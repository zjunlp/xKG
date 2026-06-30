"""
CodeVerifier: 自动验证代码片段的可执行性和语法正确性

核心功能：
1. 自动提取代码片段中的依赖包
2. 在Docker环境中自动安装依赖并运行代码
3. 自动捕获报错并进行debug修复
4. 验证语法和可执行性，输出最终可运行代码和依赖
"""

import os
import json
import ast
import re
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import docker
from docker.errors import DockerException

from .base_tool import BaseTool
from ..schema.garph import VerifiableCodeBlock, Node, Technique, CodeBlock
from ..utils import get_config, get_code_rag_config, sanitize_filename, get_process_path, should_save_process
from ..llm import extract_backtick_text, extract_object
from ..llm.prompt import CODE_REPAIR_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    is_executable: bool
    is_syntactically_correct: bool
    fixed_code: Optional[str]
    final_requirements: List[str]
    error_messages: List[str]
    execution_output: Optional[str]
    debug_attempts: int


class DependencyExtractor:
    """Extract dependencies from Python code"""

    # Mapping from Python import names to pip package names
    IMPORT_TO_PIP = {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'bs4': 'beautifulsoup4',
        'yaml': 'pyyaml',
        'attr': 'attrs',
        'dotenv': 'python-dotenv',
        'gi': 'pygobject',
        'wx': 'wxPython',
        'skimage': 'scikit-image',
        'serial': 'pyserial',
        'usb': 'pyusb',
        'git': 'gitpython',
        'lxml': 'lxml',
        'Bio': 'biopython',
        'Crypto': 'pycryptodome',
        'dateutil': 'python-dateutil',
        'jose': 'python-jose',
        'jwt': 'PyJWT',
        'magic': 'python-magic',
        'mpl_toolkits': 'matplotlib',
        'sentence_transformers': 'sentence-transformers',
    }

    # Packages that cannot be pip-installed (skip them)
    SKIP_PACKAGES = {'typing', 'typing_extensions', 'types', 'abc', 'dataclasses'}

    def __init__(self):
        self.stdlib_modules = self._load_stdlib_modules()
        self.common_packages = set(self._load_common_packages())

    def _sanitize_code_for_imports(self, code: str) -> str:
        """Remove markdown code fence markers from code"""
        sanitized_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            sanitized_lines.append(line)
        return "\n".join(sanitized_lines)

    def _load_stdlib_modules(self) -> set:
        """Load Python standard library module names"""
        import sys
        stdlib_set = set()
        try:
            stdlib_set.update(sys.builtin_module_names)
        except:
            pass

        stdlib_list = {
            'json', 'os', 'sys', 're', 'ast', 'collections', 'itertools', 'functools',
            'operator', 'copy', 'pickle', 'struct', 'hashlib', 'hmac', 'uuid', 'random',
            'math', 'cmath', 'decimal', 'fractions', 'statistics',
            'datetime', 'time', 'calendar', 'locale', 'gettext',
            'io', 'codecs', 'string', 'textwrap', 'unicodedata', 'stringprep',
            'readline', 'rlcompleter', 'difflib',
            'enum', 'numbers', 'types', 'copyreg', 'pprint', 'reprlib', 'weakref',
            'pathlib', 'fileinput', 'stat', 'filecmp', 'tempfile', 'glob', 'fnmatch',
            'linecache', 'shutil',
            'shelve', 'marshal', 'dbm', 'sqlite3',
            'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',
            'csv', 'configparser', 'netrc', 'xdrlib', 'plistlib', 'secrets',
            'errno', 'ctypes', 'platform', 'getopt', 'getpass',
            'threading', 'multiprocessing', 'concurrent', 'subprocess', 'sched', 'queue',
            'select', 'selectors', 'asyncio', 'socket', 'ssl', 'email', 'http', 'urllib',
            'html', 'xml',
            'tkinter', 'turtledemo', 'webbrowser',
            'pydoc', 'doctest', 'unittest', 'test', 'bdb', 'pdb', 'profile', 'pstats',
            'timeit', 'trace', 'tracemalloc',
            'gc', 'inspect', 'site', 'sysconfig', 'importlib', 'imp',
            'builtins', '__builtin__', '__future__'
        }
        stdlib_set.update(stdlib_list)
        return stdlib_set

    def _is_stdlib(self, package_name: str) -> bool:
        """Check if package is in standard library"""
        return package_name in self.stdlib_modules

    def _load_common_packages(self) -> List[str]:
        """Load common third-party package names"""
        return [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'sklearn',
            'torch', 'tensorflow', 'keras', 'transformers', 'datasets',
            'sentence_transformers', 'torchvision', 'torchaudio',
            'requests', 'beautifulsoup4', 'pillow', 'opencv-python',
            'plotly', 'bokeh', 'dash', 'flask', 'django', 'fastapi',
            'nltk', 'spacy', 'scikit-learn', 'xgboost', 'lightgbm',
            'statsmodels', 'sympy', 'networkx', 'openai', 'tqdm'
        ]

    def _map_import_to_pip(self, import_name: str) -> str:
        """Map Python import name to pip package name"""
        # Direct mapping
        if import_name in self.IMPORT_TO_PIP:
            return self.IMPORT_TO_PIP[import_name]
        # Return original name if no mapping
        return import_name

    def extract_from_imports(self, code: str) -> List[str]:
        """Extract non-stdlib dependencies from code"""
        dependencies = set()
        code_for_parsing = self._sanitize_code_for_imports(code)

        try:
            tree = ast.parse(code_for_parsing)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        package = alias.name.split('.')[0].strip()
                        if package and not self._is_stdlib(package) and package not in self.SKIP_PACKAGES:
                            pip_name = self._map_import_to_pip(package)
                            dependencies.add(pip_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        package = node.module.split('.')[0].strip()
                        if package and not self._is_stdlib(package) and package not in self.SKIP_PACKAGES:
                            pip_name = self._map_import_to_pip(package)
                            dependencies.add(pip_name)
        except SyntaxError:
            # Fallback to regex-based extraction
            from_import_pattern = re.compile(r'^from\s+([A-Za-z_][\w\.]*)\s+import\b')
            import_pattern = re.compile(r'^import\s+(.+)$')
            for raw_line in code_for_parsing.splitlines():
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('"""') or line.startswith("'''"):
                    continue
                match = from_import_pattern.match(line)
                if match:
                    package = match.group(1).split('.')[0]
                    if package and not self._is_stdlib(package) and package not in self.SKIP_PACKAGES:
                        pip_name = self._map_import_to_pip(package)
                        dependencies.add(pip_name)
                    continue
                match = import_pattern.match(line)
                if match:
                    modules_part = match.group(1)
                    modules_part = modules_part.split('#', 1)[0]
                    for module_entry in modules_part.split(','):
                        module_entry = module_entry.strip()
                        if not module_entry:
                            continue
                        module_name = module_entry.split()[0]
                        module_name = module_name.split('.')[0]
                        if module_name and not self._is_stdlib(module_name) and module_name not in self.SKIP_PACKAGES:
                            pip_name = self._map_import_to_pip(module_name)
                            dependencies.add(pip_name)

        valid_deps = []
        for dep in dependencies:
            if dep and len(dep) > 0 and all(c.isalnum() or c in '-_.' for c in dep):
                valid_deps.append(dep)

        return sorted(valid_deps)

    def generate_requirements_txt(self, dependencies: List[str]) -> str:
        """Generate requirements.txt content from dependency list"""
        return "\n".join(dependencies)


class EnvironmentManager:
    """Manage Docker environments for code execution"""

    def __init__(self, docker_image: str = "xkg-coderunner", article_id: str = None):
        self.docker_image = docker_image
        self.article_id = article_id or "default"
        self.client = None
        self.container_cache = {}
        self.persistent_container = None

        project_root = Path(__file__).resolve().parents[3]
        self.base_requirements_file = project_root / "requirements.txt"

        try:
            self.client = docker.from_env()
            self.client.ping()
        except (DockerException, Exception) as e:
            logger.warning(f"Docker not available ({e}), falling back to local environment")
            self.client = None

    def create_docker_container(self, requirements_content: str) -> Optional[str]:
        """Create Docker container with specified requirements"""
        if not self.client:
            return None

        import hashlib
        content_hash = hashlib.md5(requirements_content.encode()).hexdigest()[:8]
        safe_article_id = self.article_id.lower().replace('_', '-')
        cache_key = f"{safe_article_id}-{content_hash}"

        if cache_key in self.container_cache:
            logger.info(f"Using cached Docker image: {cache_key}")
            return self.container_cache[cache_key]

        try:
            self.client.images.get(self.docker_image)
            logger.info(f"Found existing base image: {self.docker_image}")

            article_image_tag = f"{self.docker_image}-{safe_article_id}"

            try:
                self.client.images.get(article_image_tag)
                logger.info(f"Using existing article-specific image: {article_image_tag}")
            except docker.errors.ImageNotFound:
                logger.info(f"Creating article-specific copy: {article_image_tag}")
                base_image = self.client.images.get(self.docker_image)
                base_image.tag(article_image_tag)
                logger.info(f"Created article-specific image: {article_image_tag}")

            if not self.persistent_container:
                self.persistent_container = self.create_persistent_container(article_image_tag)

            self.container_cache[cache_key] = article_image_tag
            return article_image_tag

        except Exception as e:
            logger.info(f"Base image {self.docker_image} not found: {e}")

        return None

    def run_in_persistent_container(self, code: str, container_name: str, timeout: int = 180, requirements: List[str] = None, code_block_name: str = "unknown") -> Tuple[bool, str, str]:
        """Run code in persistent Docker container"""
        if not self.client:
            return False, "", "Docker not available"

        try:
            container = self.client.containers.get(container_name)

            if requirements:
                logger.info(f"Installing requirements: {requirements}")
                install_cmd = f"pip install --timeout 300 --retries 3 --no-cache-dir --break-system-packages --index-url https://pypi.tuna.tsinghua.edu.cn/simple {' '.join(requirements)}"
                logger.debug("Starting pip install (this may take a while)...")

                exec_result = container.exec_run(["bash", "-c", install_cmd])

                if exec_result.exit_code != 0:
                    output = exec_result.output.decode()
                    logger.error(f"Install failed: {output}")
                    return False, "", f"Failed to install requirements: {output}"

            import time
            timestamp = int(time.time())
            safe_article_id = self.article_id.replace('_', '-').replace(' ', '-')
            safe_code_block_name = code_block_name.replace(' ', '-').replace('(', '').replace(')', '').replace(':', '')
            test_file_path = f"/tmp/test_{safe_article_id}_{safe_code_block_name}_{timestamp}.py"

            logger.debug(f"Writing code to {test_file_path}")
            # Use base64 encoding with Python (avoids bash quoting issues)
            import base64
            b64_code = base64.b64encode(code.encode()).decode('ascii')
            # Pass base64 string as an argument to Python script
            write_python_cmd = (
                f"import base64, sys; "
                f"code = base64.b64decode('{b64_code}'); "
                f"open('{test_file_path}', 'wb').write(code)"
            )
            exec_result = container.exec_run(["python", "-c", write_python_cmd])

            if exec_result.exit_code != 0:
                error_msg = exec_result.output.decode()
                logger.error(f"Failed to create test file: {error_msg}")
                return False, "", f"Failed to create test file: {error_msg}"

            logger.debug(f"Running test file: {test_file_path}")
            exec_result = container.exec_run(["python", test_file_path])

            container.exec_run(["rm", "-f", test_file_path])

            # Combine stdout and stderr
            output = exec_result.output.decode('utf-8', errors='replace')

            if exec_result.exit_code == 0:
                logger.info(f"Execution successful, output: {output[:100]}")
                return True, output, ""
            else:
                logger.error(f"Execution failed with exit code {exec_result.exit_code}: {output[:200]}")
                return False, "", output

        except Exception as e:
            return False, "", str(e)

    def run_locally(self, code: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """Run code in local Python environment"""
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Execution timeout"
        except Exception as e:
            return False, "", str(e)

    def create_persistent_container(self, image_tag: str) -> Optional[str]:
        """Create persistent Docker container"""
        if not self.client:
            return None

        container_name = f"code_verifier_{self.article_id}"

        try:
            try:
                existing_container = self.client.containers.get(container_name)
                if existing_container.status == 'running':
                    logger.info(f"Found existing running container: {container_name}")
                    return container_name
                else:
                    logger.info(f"Starting existing container: {container_name}")
                    existing_container.start()
                    return container_name
            except docker.errors.NotFound:
                logger.info(f"Creating new persistent container: {container_name}")
                container = self.client.containers.create(
                    image_tag,
                    command=["bash", "-c", "tail -f /dev/null"],
                    detach=True,
                    name=container_name
                )

                container.start()
                logger.info(f"Created new persistent container: {container.name}")
                return container.name

        except Exception as e:
            logger.warning(f"Failed to create persistent container: {e}")
            return None

    def cleanup_persistent_container(self) -> bool:
        """Cleanup persistent Docker container"""
        if not self.client:
            return False

        container_name = f"code_verifier_{self.article_id}"

        try:
            container = self.client.containers.get(container_name)
            logger.info(f"Cleaning up persistent container: {container_name}")

            if container.status == 'running':
                container.stop(timeout=10)
                logger.info(f"Stopped container: {container_name}")

            container.remove()
            logger.info(f"Removed container: {container_name}")

            self.persistent_container = None
            return True
        except docker.errors.NotFound:
            logger.info(f"Container {container_name} not found, already cleaned up")
            self.persistent_container = None
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup container {container_name}: {e}")
            return False


class CodeExecutor:
    """Execute and debug Python code"""

    def __init__(self, environment_manager: EnvironmentManager, llm_backend=None, model: str = None):
        self.env_manager = environment_manager
        self.llm_backend = llm_backend
        self.model = model
        self.max_debug_attempts = 10

    def _check_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _extract_error_info(self, error: str) -> Tuple[Optional[int], str]:
        """Extract line number and error type from error message"""
        line_number = None
        error_type = "unknown"

        patterns = [
            r'line (\d+)',
            r'File "<string>", line (\d+)',
            r'(\d+):\d+:',
        ]
        for pattern in patterns:
            match = re.search(pattern, error)
            if match:
                try:
                    line_number = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue

        if "SyntaxError" in error:
            error_type = "syntax"
        elif "ModuleNotFoundError" in error or "ImportError" in error:
            error_type = "import"
        elif "IndentationError" in error:
            error_type = "indentation"
        elif "NameError" in error:
            error_type = "name"
        elif "AttributeError" in error:
            error_type = "attribute"
        elif "TypeError" in error:
            error_type = "type"

        return line_number, error_type

    def _attempt_fix_with_llm(self, code: str, error: str, dependencies: List[str]) -> Tuple[str, List[str]]:
        """Attempt to fix code using LLM"""
        if not self.llm_backend:
            return code, []

        try:
            # System prompt with detailed debugging instructions
            system_prompt = CODE_REPAIR_PROMPT

            # User prompt with specific error context
            error_truncated = error[:500] if len(error) > 500 else error
            user_prompt = f"""CRITICAL: YOU MUST MODIFY THE CODE BASED ON THE ERROR!

ERROR MESSAGE:
{error_truncated}

BROKEN CODE THAT NEEDS FIXING:
```python
{code}
```

INSTRUCTIONS:
1. Read the error message carefully
2. Identify the exact issue causing the error
3. Apply fixes to resolve the error
4. DO NOT return the same code - you must make changes
5. Return ONLY the fixed Python code, no explanations or markdown"""

            # Use code_model for code fixing
            code_model = get_config().get('code_model', self.model)

            # Call LLM backend with system and user prompts
            response_tuple = self.llm_backend.query(
                user_message=user_prompt,
                model=code_model,
                system_message=system_prompt,
                temperature=0.1,
                max_tokens=2000
            )
            # Extract the response text
            fixed_code = response_tuple[0].strip() if isinstance(response_tuple, tuple) else response_tuple.strip()

            # Remove markdown code fence markers
            if fixed_code.startswith("```python"):
                fixed_code = fixed_code[9:]
            if fixed_code.startswith("```"):
                fixed_code = fixed_code[3:]
            if fixed_code.endswith("```"):
                fixed_code = fixed_code[:-3]
            fixed_code = fixed_code.strip()

            # Check if code changed and syntax is correct
            if fixed_code != code and self._check_syntax(fixed_code):
                logger.info("LLM fix successful")
                # Extract new dependencies
                extractor = DependencyExtractor()
                new_deps = extractor.extract_from_imports(fixed_code)
                new_deps = [d for d in new_deps if d not in dependencies]
                return fixed_code, new_deps

            logger.warning("LLM fix failed or produced same code")
            return code, []

        except Exception as e:
            logger.warning(f"LLM fix exception: {e}")
            return code, []

    def execute_code(self, code: str, requirements: List[str], use_docker: bool = True, code_block_name: str = "unknown") -> VerificationResult:
        """Execute code with iterative debugging"""
        dependencies = requirements.copy()
        current_code = code
        debug_attempts = 0
        error_messages = []
        last_output = ""

        while debug_attempts < self.max_debug_attempts:
            requirements_content = "\n".join(dependencies)

            if use_docker and self.env_manager.client:
                image_tag = self.env_manager.create_docker_container(requirements_content)
                if image_tag and self.env_manager.persistent_container:
                    success, stdout, stderr = self.env_manager.run_in_persistent_container(
                        current_code, self.env_manager.persistent_container, requirements=dependencies, code_block_name=code_block_name
                    )
                else:
                    success, stdout, stderr = self.env_manager.run_locally(current_code)
            else:
                success, stdout, stderr = self.env_manager.run_locally(current_code)

            # Check for actual success:
            # - exit_code == 0 (success == True)
            # - NO error messages in stderr (stderr is empty or no Traceback/Error)
            # stdout can be empty (some code doesn't produce output)
            has_error = stderr and ("Traceback" in stderr or "Error" in stderr or "error" in stderr.lower())
            is_truly_executable = success and not has_error

            logger.info(f"Attempt {debug_attempts + 1}: success={success}, has_error={has_error}, stderr_len={len(stderr)}, stdout_len={len(stdout)}")
            if stderr:
                logger.debug(f"  stderr: {stderr[:100]}")
            if stdout:
                logger.debug(f"  stdout: {stdout[:100]}")

            if is_truly_executable:
                logger.info(f"✓ Code execution successful")
                return VerificationResult(
                    is_executable=True,
                    is_syntactically_correct=True,
                    fixed_code=current_code,
                    final_requirements=dependencies,
                    error_messages=error_messages,
                    execution_output=stdout,
                    debug_attempts=debug_attempts
                )

            last_output = stderr if stderr else stdout
            error_messages.append(f"Attempt {debug_attempts + 1}: {last_output[:150]}")

            # Try LLM fix
            fixed_code, new_deps = self._attempt_fix_with_llm(current_code, last_output, dependencies)

            code_changed = fixed_code != current_code
            deps_changed = len(new_deps) > 0

            if code_changed:
                logger.info(f"Fix attempt {debug_attempts + 1}: code modified by LLM")
                current_code = fixed_code

            if deps_changed:
                logger.info(f"Adding {len(new_deps)} new dependencies: {new_deps}")
                dependencies.extend(new_deps)
                dependencies = list(set(dependencies))

            # Stop if no changes
            if not code_changed and not deps_changed:
                logger.info(f"Fix attempt {debug_attempts + 1}: no changes applied, stopping debug loop")
                break

            debug_attempts += 1

        # Final attempt with current code
        logger.info("Returning last attempted code")
        return VerificationResult(
            is_executable=False,
            is_syntactically_correct=self._check_syntax(current_code),
            fixed_code=current_code,
            final_requirements=dependencies,
            error_messages=error_messages,
            execution_output=last_output if last_output else None,
            debug_attempts=debug_attempts
        )


class CodeVerifier(BaseTool):
    """
    Main CodeVerifier component - verifies and fixes code blocks in Node objects.

    Interface:
        - verify(node: Node, save_process: bool = None) -> Node
          Input: unverified Node with code blocks
          Output: verified Node with fixed/executable code
    """

    def __init__(
        self,
        model: str = None,
        memory: Optional[str] = None,
        use_docker: bool = True,
        article_id: str = None,
        docker_image: str = None,
        max_debug_attempts: int = None,
    ):
        # Use code_model from config if model not provided
        if model is None:
            model = get_config().get('code_model', get_config().get('model'))

        super().__init__(model, memory)

        # Use config docker_image if not provided
        if docker_image is None:
            docker_image = get_code_rag_config().get('docker_image', 'xkg-coderunner:latest')

        # Use config max_debug_attempts if not provided
        if max_debug_attempts is None:
            max_debug_attempts = get_code_rag_config().get('max_debug_attempts', 8)

        self.dependency_extractor = DependencyExtractor()
        self.env_manager = EnvironmentManager(docker_image=docker_image, article_id=article_id)
        # Pass LLM backend to CodeExecutor for intelligent repair
        self.code_executor = CodeExecutor(self.env_manager, llm_backend=self.backend, model=self.model)
        self.code_executor.max_debug_attempts = max_debug_attempts
        self.use_docker = use_docker
        self.article_id = article_id or "default"
        self.max_debug_attempts = max_debug_attempts

        logger.info(f"CodeVerifier initialization completed with model={self.model}, docker_image={docker_image}, max_debug_attempts={max_debug_attempts}")

    def _get_all_techniques(self, tech: Technique) -> List[Technique]:
        """Recursively collect all technique nodes including nested components"""
        nodes = [tech]
        if tech.components:
            for component in tech.components:
                nodes.extend(self._get_all_techniques(component))
        return nodes

    def verify_code_block(self, code_block: VerifiableCodeBlock, skip_dependency_install: bool = False) -> VerifiableCodeBlock:
        """Verify a single code block"""
        logger.info(f"Starting verification of code block for article {self.article_id}...")

        logger.info("1. Extracting dependency packages...")
        code_deps = self.dependency_extractor.extract_from_imports(code_block.implementation)
        if code_block.test:
            test_deps = self.dependency_extractor.extract_from_imports(code_block.test)
            code_deps.extend(test_deps)

        all_deps = list(set(code_deps + code_block.package))
        logger.info(f"Extracted dependencies: {all_deps}")

        # Prepare the code to verify
        # If there's a test block, combine implementation + test for verification
        # This ensures the test block is actually executed and verified
        verify_code = code_block.implementation
        if code_block.test:
            verify_code = code_block.implementation + "\n\n" + code_block.test
            logger.info("2. Will verify combined implementation + test code")
        else:
            logger.info("2. Will verify implementation code only")

        logger.info("3. Executing code verification with debug loop...")
        if skip_dependency_install:
            logger.info("Skipping dependency installation - using pre-installed environment")
            result = self.code_executor.execute_code(
                code=verify_code,
                requirements=[],
                use_docker=self.use_docker,
                code_block_name=code_block.documentation[:50] if code_block.documentation else "test-code"
            )
        else:
            result = self.code_executor.execute_code(
                code=verify_code,
                requirements=all_deps,
                use_docker=self.use_docker,
                code_block_name=code_block.documentation[:50] if code_block.documentation else "test-code"
            )

        verified_code_block = VerifiableCodeBlock(
            implementation=result.fixed_code or code_block.implementation,
            test=code_block.test,
            documentation=code_block.documentation,
            package=result.final_requirements,
            language=code_block.language,
            is_executable=result.is_executable,
            is_syntactically_correct=result.is_syntactically_correct,
            error_messages=result.error_messages,
            debug_attempts=result.debug_attempts,
            execution_output=result.execution_output,
            final_requirements=result.final_requirements,
            fixed_code=result.fixed_code or code_block.implementation
        )

        verification_info = f"\n\n=== Verification Information ===\n"
        verification_info += f"Verification status: {'Executable' if result.is_executable else 'Not executable'}\n"
        verification_info += f"Syntax correct: {'Yes' if result.is_syntactically_correct else 'No'}\n"
        verification_info += f"Debug attempts: {result.debug_attempts}\n"
        if result.execution_output:
            verification_info += f"Execution output: {result.execution_output[:200]}\n"
        if result.error_messages:
            verification_info += f"Error messages: {'; '.join(result.error_messages[:3])}\n"

        verified_code_block.documentation += verification_info

        logger.info(f"Verification completed, final dependencies: {verified_code_block.package}")
        logger.info(f"Final code executable: {verified_code_block.is_executable}")

        if self.env_manager.persistent_container:
            logger.info("Cleaning up persistent container after verification...")
            self.env_manager.cleanup_persistent_container()

        return verified_code_block

    def verify(
        self,
        node: Node,
        save_process: bool = None,
        use_parallel: bool = True,
        max_workers: int = 3,
        max_debug_attempts: int = None,
        skip_dependency_install: bool = False,
    ) -> Node:
        """
        Verify code blocks in Node object.

        Args:
            node: Unverified Node object
            save_process: Whether to save intermediate results to process directory
            use_parallel: Whether to parallelize code block verification
            max_workers: Maximum number of parallel workers
            max_debug_attempts: Maximum number of debug/fix attempts per code block (None = use default from config)
            skip_dependency_install: Skip installing dependencies in Docker

        Returns:
            Verified Node object with fixed code
        """
        try:
            # Use provided max_debug_attempts or fall back to instance value
            if max_debug_attempts is not None:
                self.code_executor.max_debug_attempts = max_debug_attempts

            logger.info(f"Starting code verification for node: {node.paper_title}...")
            logger.info(f"  Config: max_debug_attempts={self.code_executor.max_debug_attempts}, skip_dependency_install={skip_dependency_install}")

            # Recursively collect all technique nodes
            all_techniques = []
            for tech in node.techniques:
                all_techniques.extend(self._get_all_techniques(tech))

            # Filter techniques with code
            techniques_with_code = [
                tech for tech in all_techniques
                if tech.code and isinstance(tech.code, CodeBlock)
            ]

            if not techniques_with_code:
                logger.info("No code blocks to verify in this node.")
                return node

            logger.info(f"Found {len(techniques_with_code)} code blocks to verify")

            # Verify code blocks (optionally in parallel)
            verifiable_blocks = []
            if use_parallel and len(techniques_with_code) > 1:
                logger.info(f"Verifying code blocks in parallel with {max_workers} workers...")
                verifiable_blocks = self._verify_code_blocks_parallel(techniques_with_code, max_workers, skip_dependency_install)
            else:
                logger.info("Verifying code blocks sequentially...")
                for i, tech in enumerate(techniques_with_code):
                    vblock = VerifiableCodeBlock(
                        implementation=tech.code.implementation,
                        test=tech.code.test,
                        documentation=tech.code.documentation,
                        package=tech.code.package,
                        language='python'
                    )
                    logger.info(f"Verifying code block {i+1}/{len(techniques_with_code)}: {tech.name}")
                    verified_vblock = self.verify_code_block(vblock, skip_dependency_install=skip_dependency_install)
                    verifiable_blocks.append((tech, verified_vblock))

            # Update techniques with verified code
            for tech, verified_vblock in verifiable_blocks:
                tech.code = CodeBlock(
                    implementation=verified_vblock.fixed_code or verified_vblock.implementation,
                    test=verified_vblock.test,
                    documentation=verified_vblock.documentation,
                    package=verified_vblock.final_requirements or verified_vblock.package
                )

            logger.info(f"Code verification completed for {len(verifiable_blocks)} blocks")

            # Filter out techniques with code=None
            node_dict = asdict(node)
            node_dict = self._filter_null_code_techniques(node_dict)
            node = Node.from_dict(node_dict)
            logger.info(f"Filtered techniques with code=None, remaining: {len(node.techniques)}")

            # Save verification results
            should_save = save_process if save_process is not None else should_save_process()
            if should_save:
                self._save_verification_results(
                    f"{sanitize_filename(node.paper_title)}_verified",
                    [vb for _, vb in verifiable_blocks]
                )

            return node

        except Exception as e:
            logger.error(f"Error during code verification: {e}", exc_info=True)
            return node

    def _verify_code_blocks_parallel(
        self,
        techniques: List[Technique],
        max_workers: int,
        skip_dependency_install: bool = False
    ) -> List[Tuple[Technique, VerifiableCodeBlock]]:
        """Verify code blocks in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for i, tech in enumerate(techniques):
                vblock = VerifiableCodeBlock(
                    implementation=tech.code.implementation,
                    test=tech.code.test,
                    documentation=tech.code.documentation,
                    package=tech.code.package,
                    language='python'
                )
                future = executor.submit(self.verify_code_block, vblock, skip_dependency_install=skip_dependency_install)
                futures[future] = (i, tech)

            for future in as_completed(futures):
                idx, tech = futures[future]
                try:
                    verified_vblock = future.result()
                    results.append((tech, verified_vblock))
                    logger.info(f"Completed verification for block {idx+1}: {tech.name}")
                except Exception as e:
                    logger.error(f"Error verifying block {idx+1} ({tech.name}): {e}")
                    # Return original code block on error
                    verified_vblock = VerifiableCodeBlock(
                        implementation=tech.code.implementation,
                        test=tech.code.test,
                        documentation=tech.code.documentation,
                        package=tech.code.package,
                        language='python',
                        is_executable=False,
                        error_messages=[str(e)]
                    )
                    results.append((tech, verified_vblock))

        return results

    def _filter_null_code_techniques(self, node_dict: Dict) -> Dict:
        """Recursively filter out techniques with code=None"""
        def filter_techniques(techniques):
            filtered = []
            for tech in techniques:
                if tech.get('code') is not None:
                    # Recursively filter components
                    if tech.get('components'):
                        tech['components'] = filter_techniques(tech['components'])
                    filtered.append(tech)
            return filtered

        if node_dict.get('techniques'):
            node_dict['techniques'] = filter_techniques(node_dict['techniques'])

        return node_dict

    def _save_verification_results(self, filename_prefix: str, verified_blocks: List[VerifiableCodeBlock]):
        """Save verification results to process cache directory"""
        try:
            process_base = get_process_path()
            output_dir = process_base / "code_verifier"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save verification results as JSON
            output_file = output_dir / f"{filename_prefix}_blocks.json"

            verified_data = [asdict(block) for block in verified_blocks]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(verified_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Verification results saved to {output_file}")
        except Exception as e:
            logger.warning(f"Failed to save verification results: {e}")

    def cleanup(self):
        """Cleanup resources"""
        if self.env_manager:
            self.env_manager.cleanup_persistent_container()
