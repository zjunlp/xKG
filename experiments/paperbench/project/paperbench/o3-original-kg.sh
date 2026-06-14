nohup: ignoring input
#0 building with "default" instance using docker driver

#1 [internal] load build definition from Dockerfile
#1 transferring dockerfile: 342B 0.0s done
#1 DONE 0.2s

#2 [internal] load metadata for docker.io/library/aisi-basic-agent:latest
#2 DONE 0.0s

#3 [internal] load .dockerignore
#3 transferring context:
#3 transferring context: 67B done
#3 DONE 0.2s

#4 [1/2] FROM docker.io/library/aisi-basic-agent:latest
#4 DONE 0.0s

#5 [internal] load build context
#5 transferring context: 48.12kB 0.0s done
#5 DONE 0.2s

#4 [1/2] FROM docker.io/library/aisi-basic-agent:latest
#4 CACHED

#6 [2/2] COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
#6 DONE 1.5s

#7 exporting to image
#7 exporting layers
#7 exporting layers 0.7s done
#7 writing image sha256:097d35a43468f1b335b212098019344ddc9a4559c2148358c6f1bce79acf777f 0.1s done
#7 naming to docker.io/library/aisi-basic-agent-custom:latest 0.1s done
#7 DONE 1.0s
[1763581305.1178458] PROBE: --- paperbench/nano/entrypoint.py: File imported by Python. ---
[1763581306.764462] PROBE: --- All modules imported successfully. ---
[1763581306.7644918] PROBE: --- DefaultRunnerArgs class is being defined. ---
[1763581306.7650707] PROBE: --- __name__ == '__main__': Script is being run directly. ---
[1763581306.7650857] PROBE: --- Preparing to call chz.entrypoint(main)... ---
[1763581306.8289697] PROBE: --- paperbench/agents/utils.py: prepare_agent_dir_config() started. ---
[1763581306.8294299] PROBE: PaperBench.run_group_id property is being accessed/initialized.
[1763581306.8304975] PROBE: --- chz.entrypoint(main) created. ---
[1763581306.830518] PROBE: --- Preparing to call nanoeval_entrypoint(...). This will start the event loop. ---
[1763581306.8309567] PROBE: --- async main() function started. ---
[1763581306.8309708] PROBE: --- Calling run_sanity_checks()... ---
[1763581306.868208] PROBE: --- run_sanity_checks() finished. ---
[1763581306.868243] PROBE: --- Preparing to call await run(...). This is the main evaluation step. ---
[1763581306.873585] PROBE: PaperBench.get_instances() called.
[1763581306.873606] PROBE: PaperBench.get_instances(): GRADER_OPENAI_API_KEY check passed.
[2m2025-11-19T19:41:46.873750Z[0m [[32m[1minfo     [0m] [1m[1;35mWriting run group logs to /disk/disk_20T/luoyujie/preparedness/project/paperbench/runs/2025-11-19T19-41-46-GMT_run-group_aisi-basic-agent-my-o3/group.log[0m[0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m
[1763581306.8740354] PROBE: PaperBench.get_instances(): Reading paper split file: /disk/disk_20T/luoyujie/preparedness/project/paperbench/experiments/splits/dev.txt
[1763581306.874094] PROBE: PaperBench.get_instances(): Found 2 paper_ids from split file.
[1763581306.8741195] PROBE: PaperBench.get_instances(): Got agents_dir: /disk/disk_20T/luoyujie/preparedness/project/paperbench/paperbench/agents.
[1763581306.8741345] PROBE: PaperBench.get_instances(): Entering loop to create tasks...
[1763581306.8741443] PROBE: PaperBench.get_instances(): Loop 1: attempt_idx = 0.
[1763581306.8741505] PROBE: PaperBench.get_instances(): Loop 2: paper_id = sample-specific-masks.
[1763581306.8755653] PROBE: PaperBench.get_instances(): Finished processing for paper_id sample-specific-masks.
[1763581306.8755853] PROBE: PaperBench.get_instances(): Loop 2: paper_id = bridging-data-gaps.
[1763581306.8767443] PROBE: PaperBench.get_instances(): Finished processing for paper_id bridging-data-gaps.
[1763581306.8767614] PROBE: PaperBench.get_instances(): Finished loop for creating tasks.
[2m2025-11-19T19:41:46.876781Z[0m [[32m[1minfo     [0m] [1mPreparing to run 2 tasks...   [0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m
[1763581306.8772278] PROBE: PaperBench.get_instances(): Returning {len(tasks)} tasks.

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[2m2025-11-19T19:41:46.952198Z[0m [[32m[1minfo     [0m] [1m[36m[1mEvalboard Monitor: unknown, add in pr?[0m[0m [36mcomponent[0m=[35mnanoeval.evaluation[0m
[2m2025-11-19T19:41:46.952400Z[0m [[33m[1mwarning  [0m] [1mUnable to find model name, using fallback name='nanoeval'[0m [36mcomponent[0m=[35mnanoeval.eval[0m

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[2m2025-11-19T19:41:46.952946Z[0m [[33m[1mwarning  [0m] [1mNo slack integration configured, so not sending user notification[0m [36mcomponent[0m=[35mnanoeval.library_config[0m

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[2m2025-11-19T19:41:46.953372Z[0m [[33m[1mwarning  [0m] [1mUnable to find model name, using fallback name='nanoeval'[0m [36mcomponent[0m=[35mnanoeval.eval[0m
[2m2025-11-19T19:41:46.953523Z[0m [[33m[1mwarning  [0m] [1mNo slack integration configured, so not using slack-tqdm[0m [36mcomponent[0m=[35mnanoeval.library_config[0m
asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:00<?, ?it/s]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:00<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[1763581306.9734144] PROBE: ExternalPythonCodingSolver.run() called.
[1763581306.9734352] PROBE: ExternalPythonCodingSolver.run() started for task sample-specific-masks_6227908e-20d5-4a28-9fa6-73174810fbab.
[1763581306.9734404] PROBE: ExternalPythonCodingSolver.run(): Entering 'async with self._start_computer(task)' block.
[1763581306.9734507] PROBE: ExternalPythonCodingSolver._start_computer() started for task sample-specific-masks_6227908e-20d5-4a28-9fa6-73174810fbab.
[1763581306.99596] PROBE: ExternalPythonCodingSolver._start_computer(): Calling alcatraz_config.build(). This will start the Docker container.

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[1763581307.0276713] PROBE: ExternalPythonCodingSolver.run() called.
[1763581307.0277078] PROBE: ExternalPythonCodingSolver.run() started for task bridging-data-gaps_c72731e2-1eb4-4184-a416-4d1a615928b1.
[1763581307.0277193] PROBE: ExternalPythonCodingSolver.run(): Entering 'async with self._start_computer(task)' block.
[1763581307.0277398] PROBE: ExternalPythonCodingSolver._start_computer() started for task bridging-data-gaps_c72731e2-1eb4-4184-a416-4d1a615928b1.
[1763581307.0642157] PROBE: ExternalPythonCodingSolver._start_computer(): Calling alcatraz_config.build(). This will start the Docker container.

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:01<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:01<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:02<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:02<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:03<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:03<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:04<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:04<?, ?it/s, corr=0, errs=0, fail=0][1763581311.1265192] PROBE: ExternalPythonCodingSolver.run(): Computer started successfully. Now running task.setup().

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[1763581311.358934] PROBE: ExternalPythonCodingSolver.run(): Computer started successfully. Now running task.setup().

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:05<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:05<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[2m2025-11-19T19:41:52.528321Z[0m [[33m[1mwarning  [0m] [1mContainer health not available for this container, ignoring.[0m [36mcomponent[0m=[35malcatraz.clusters.local[0m
[2m2025-11-19T19:41:52.770841Z[0m [[33m[1mwarning  [0m] [1mContainer health not available for this container, ignoring.[0m [36mcomponent[0m=[35malcatraz.clusters.local[0m

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:06<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:06<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:07<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:07<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:08<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:08<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:09<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:09<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:10<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:10<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:11<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:11<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:12<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:12<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:13<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:13<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[1763581321.010459] PROBE: ExternalPythonCodingSolver.run(): task.setup() finished. Now running self._run_agent().
[2m2025-11-19T19:42:01.010643Z[0m [[32m[1minfo     [0m] [1mAgent `aisi-basic-agent-my-o3` is attempting to replicate the `bridging-data-gaps` paper...[0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m
[2m2025-11-19T19:42:01.013440Z[0m [[32m[1minfo     [0m] [1m[1;35mWriting logs for run to /disk/disk_20T/luoyujie/preparedness/project/paperbench/runs/2025-11-19T19-41-46-GMT_run-group_aisi-basic-agent-my-o3/bridging-data-gaps_c72731e2-1eb4-4184-a416-4d1a615928b1/run.log[0m[0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:14<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:14<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:15<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:15<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:16<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:16<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:17<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:17<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:18<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:18<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:19<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:19<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[1763581327.0767193] PROBE: ExternalPythonCodingSolver.run(): task.setup() finished. Now running self._run_agent().
[2m2025-11-19T19:42:07.076821Z[0m [[32m[1minfo     [0m] [1mAgent `aisi-basic-agent-my-o3` is attempting to replicate the `sample-specific-masks` paper...[0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m
[2m2025-11-19T19:42:07.085719Z[0m [[32m[1minfo     [0m] [1m[1;35mWriting logs for run to /disk/disk_20T/luoyujie/preparedness/project/paperbench/runs/2025-11-19T19-41-46-GMT_run-group_aisi-basic-agent-my-o3/sample-specific-masks_6227908e-20d5-4a28-9fa6-73174810fbab/run.log[0m[0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:20<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:20<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:21<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:21<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:22<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:22<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:23<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:23<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:24<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:24<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:25<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:25<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:26<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:26<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:27<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:27<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:28<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:28<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:29<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:29<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:30<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:30<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:31<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:31<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:32<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:32<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:33<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:33<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:34<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:34<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:35<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:35<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:36<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:36<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:37<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:37<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:38<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:38<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:39<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:39<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:40<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:40<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:41<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:41<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:42<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:42<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:43<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:43<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:44<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:44<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:45<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:45<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:46<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:46<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:47<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:47<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:48<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:48<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:49<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:49<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:50<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:50<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:51<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:51<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:52<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:52<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:53<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:53<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:54<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:54<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:55<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:55<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:56<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:56<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:57<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:57<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:58<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:58<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:59<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [00:59<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:00<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:00<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:01<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:01<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:02<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:02<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:03<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:03<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:04<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:04<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:05<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:05<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:06<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:06<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:07<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:07<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:08<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:08<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:09<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:09<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:10<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:10<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:11<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:11<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:12<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:12<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:13<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:13<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:14<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:14<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:15<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:15<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:16<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:16<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:17<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:17<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:18<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:18<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:19<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:19<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:20<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:20<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:21<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:21<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:22<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:22<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:23<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:23<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:24<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:24<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:25<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:25<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:26<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:26<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:27<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:27<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:28<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:28<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:29<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:29<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:30<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:30<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:31<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:31<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:32<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:32<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:33<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:33<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:34<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:34<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:35<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:35<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:36<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:36<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:37<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:37<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:38<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:38<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:39<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:39<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:40<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:40<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:41<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:41<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:42<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:42<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:43<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:43<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:44<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:44<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:45<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:45<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:46<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:46<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:47<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:47<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:48<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:48<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:49<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:49<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:50<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:50<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:51<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:51<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:52<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:52<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:53<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:53<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:54<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:54<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:55<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:55<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:56<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:56<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:57<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:57<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:58<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:58<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:59<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [01:59<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:00<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:00<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:01<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:01<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:02<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:02<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:03<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:03<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:04<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:04<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:05<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:05<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:06<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:06<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:07<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:07<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:08<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:08<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:09<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:09<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:10<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:10<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:11<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:11<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:12<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:12<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:13<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:13<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:14<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:14<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:15<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:15<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:16<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:16<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:17<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:17<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:18<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:18<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:19<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:19<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:20<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:20<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:21<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:21<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:22<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:22<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:23<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:23<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:24<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:24<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:25<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:25<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:26<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:26<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:27<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:27<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:28<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:28<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:29<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:29<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:30<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:30<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:31<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:31<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:32<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:32<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:33<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:33<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:34<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:34<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:35<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:35<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:36<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:36<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:37<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:37<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:38<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:38<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:39<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:39<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:40<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:40<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:41<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:41<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:42<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:42<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:43<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:43<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:44<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:44<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:45<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:45<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:46<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:46<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:47<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:47<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:48<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:48<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:49<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:49<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:50<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:50<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:51<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:51<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:52<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:52<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:53<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:53<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:54<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:54<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:55<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:55<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:56<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:56<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:57<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:57<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:58<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:58<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:59<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [02:59<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:00<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:00<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:01<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:01<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:02<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:02<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:03<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:03<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:04<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:04<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:05<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:05<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:06<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:06<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:07<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:07<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:08<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:08<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:09<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:09<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:10<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:10<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:11<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:11<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:12<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:12<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:13<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:13<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:14<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:14<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:15<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:15<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:16<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:16<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:17<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:17<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:18<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:18<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:19<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:19<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:20<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:20<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:21<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:21<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:22<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:22<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:23<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:23<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:25<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:25<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:26<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:26<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:27<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:27<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:28<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:28<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:29<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:29<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:30<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:30<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:31<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:31<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:32<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:32<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:33<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:33<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:34<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:34<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:35<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:35<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:36<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:36<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:37<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:37<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:38<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:38<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:39<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:39<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:40<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:40<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:41<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:41<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:42<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:42<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:43<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:43<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:44<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:44<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:45<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:45<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:46<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:46<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:47<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:47<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:48<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:48<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:49<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:49<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:50<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:50<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:51<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:51<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:52<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:52<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:53<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:53<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:54<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:54<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:55<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:55<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:56<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:56<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:57<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:57<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:58<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:58<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:59<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [03:59<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:00<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:00<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:01<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:01<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:02<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:02<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:03<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:03<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:04<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:04<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:05<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:05<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:06<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:06<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:07<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:07<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:08<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:08<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:09<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:09<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:10<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:10<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:11<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:11<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:12<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:12<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:13<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:13<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:14<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:14<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:15<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:15<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:16<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:16<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:17<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:17<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:18<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:18<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:19<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:19<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:20<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:20<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:21<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:21<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:22<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:22<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:23<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:23<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:24<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:24<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:25<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:25<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:26<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:26<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:27<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:27<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:28<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:28<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:29<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:29<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:30<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:30<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:31<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:31<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:32<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:32<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:33<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:33<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:34<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:34<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:35<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:35<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:36<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:36<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:37<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:37<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:38<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:38<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:39<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:39<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:40<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:40<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:41<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:41<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:42<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:42<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:43<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:43<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:44<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:44<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:45<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:45<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:46<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:46<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:47<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:47<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:48<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:48<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:49<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:49<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:50<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:50<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:51<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:51<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:52<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:52<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:53<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:53<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:54<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:54<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:55<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:55<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:56<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:56<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:57<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:57<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:58<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:58<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:59<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [04:59<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:00<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:00<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:01<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:01<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:02<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:02<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:03<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:03<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:04<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:04<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:05<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:05<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:06<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:06<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:07<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:07<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:08<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:08<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:09<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:09<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:10<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:10<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:11<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:11<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:12<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:12<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:13<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:13<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[2m2025-11-19T19:47:01.065646Z[0m [[32m[1minfo     [0m] [1mAgent `aisi-basic-agent-my-o3` finished running for `bridging-data-gaps_c72731e2-1eb4-4184-a416-4d1a615928b1.0`![0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m
[1763581621.0701685] PROBE: ExternalPythonCodingSolver.run(): self._run_agent() finished. Now running task.grade().
[1763581621.0710552] PROBE: --- paperbench/agents/utils.py: prepare_agent_dir_config() started. ---

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:14<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:14<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:15<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:15<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[2m2025-11-19T19:47:02.619523Z[0m [[32m[1minfo     [0m] [1mGrading the submission for bridging-data-gaps_c72731e2-1eb4-4184-a416-4d1a615928b1.0...[0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:16<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:16<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:17<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:17<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:18<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:18<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:19<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:19<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:20<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:20<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:21<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:21<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:22<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:22<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:23<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:23<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:24<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:24<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:25<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:25<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:26<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:26<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:27<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:27<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:28<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:28<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:29<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:29<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:30<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:30<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:31<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:31<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:32<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:32<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:33<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:33<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:34<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:34<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:35<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:35<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:36<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:36<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:37<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:37<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:38<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:38<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:39<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:39<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:40<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:40<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:41<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:41<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:42<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:42<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:43<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:43<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:44<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:44<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:45<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:45<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:46<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:46<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:47<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:47<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:48<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:48<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:49<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:49<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:50<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:50<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:51<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:51<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:52<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:52<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:53<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:53<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:54<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:54<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:55<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:55<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:56<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:56<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:57<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:57<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:58<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:58<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:59<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [05:59<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:00<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:00<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:01<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:01<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:02<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:02<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:03<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:03<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:04<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:04<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:05<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:05<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:06<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:06<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:07<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:07<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:08<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:08<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:09<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:09<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:10<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:10<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:11<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:11<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:12<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:12<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:13<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:13<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:14<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:14<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:15<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:15<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:16<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:16<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[2m2025-11-19T19:48:04.719759Z[0m [[32m[1minfo     [0m] [1mGrading for bridging-data-gaps_c72731e2-1eb4-4184-a416-4d1a615928b1.0 finished![0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m
[2m2025-11-19T19:48:04.749063Z[0m [[32m[1minfo     [0m] [1m[1;35mGrades saved to /disk/disk_20T/luoyujie/preparedness/project/paperbench/runs/2025-11-19T19-41-46-GMT_run-group_aisi-basic-agent-my-o3/bridging-data-gaps_c72731e2-1eb4-4184-a416-4d1a615928b1/pb_result.json[0m[0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:17<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:17<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:18<?, ?it/s, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):   0%|          | 0/2 [06:18<?, ?it/s, corr=0, errs=0, fail=0]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[1763581686.1520422] PROBE: ExternalPythonCodingSolver._start_computer() finished.

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:19<00:01,  1.01s/it, corr=0, errs=0, fail=0]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:19<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:20<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:21<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:22<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:23<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:24<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:25<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:26<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:27<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:28<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:29<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:30<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:31<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:32<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:33<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:34<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:35<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:36<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:37<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:38<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:39<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:40<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:41<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:42<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:43<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:44<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:45<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:46<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:47<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:48<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:49<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:50<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:51<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:52<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:53<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:54<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:55<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:56<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:58<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [06:59<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:00<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:01<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:02<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:03<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:04<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:05<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:06<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:07<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:08<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:09<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:10<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:11<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:12<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:13<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:14<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:15<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:16<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:17<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:18<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:19<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:20<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:21<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:22<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:23<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:24<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:25<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:26<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:27<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:28<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:29<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:30<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:31<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:32<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:33<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:34<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:35<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:36<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:37<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:38<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:39<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:40<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:41<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:42<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:43<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:44<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:45<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:46<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:47<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:48<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:49<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:50<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:51<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:52<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:53<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:54<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:55<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:56<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:57<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:58<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [07:59<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:00<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:01<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:02<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:03<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:04<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:05<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:06<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:07<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:08<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:09<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:10<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:11<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:12<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:13<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:14<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:15<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:16<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:17<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:18<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:19<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:20<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:21<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:22<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:23<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:24<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:25<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:26<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:27<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:28<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:29<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:30<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:31<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:32<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:33<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:34<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:35<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:36<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:37<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:38<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:39<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:40<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:41<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:42<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:43<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:44<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:45<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:46<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:47<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:48<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:49<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:50<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:51<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:52<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:53<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:54<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:55<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:56<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:57<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:58<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [08:59<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:00<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:01<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:02<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:03<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:04<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:05<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:06<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:07<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:08<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:09<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:10<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:11<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:12<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:13<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:14<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:15<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:16<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:17<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:18<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:19<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:20<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:21<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:22<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:23<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:24<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:25<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:26<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:27<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:28<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:29<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:30<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:31<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:32<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:33<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:34<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:35<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:36<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:37<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:38<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:39<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:40<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:41<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:42<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:43<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:44<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:45<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:46<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:47<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:48<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:49<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:50<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:51<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:52<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:53<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:54<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:55<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:56<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:57<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:58<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [09:59<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:00<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:01<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:02<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:03<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:04<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:05<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:06<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:07<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:08<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:09<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:10<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:11<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:12<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:13<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:14<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:15<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:16<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:17<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:19<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:20<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:21<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:22<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:23<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:24<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:25<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:26<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:27<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:28<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:29<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:30<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:31<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:32<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:33<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:34<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:35<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:36<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:37<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:38<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:39<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:40<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:41<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:42<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:43<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:44<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:45<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:46<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:47<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:48<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:49<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:50<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:51<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:52<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:53<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:54<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:55<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:56<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:57<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:58<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [10:59<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:00<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:01<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:02<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:03<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:04<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:05<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:06<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:07<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:08<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:09<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:10<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:11<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:12<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:13<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:14<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:15<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:16<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:17<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:18<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:19<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:20<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:21<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:22<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:23<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:24<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:25<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:26<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:27<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:28<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:29<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:30<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:31<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:32<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:33<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:34<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:35<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:36<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:37<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:38<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:39<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:40<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:41<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:42<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:43<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:44<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:45<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:46<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:47<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:48<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:49<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:50<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:51<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:52<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:53<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:54<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:55<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:56<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:57<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:58<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [11:59<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:00<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:01<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:02<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:03<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:04<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:05<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:06<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:07<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:08<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:09<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:10<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:11<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:12<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:13<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:14<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:15<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:16<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:17<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:18<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:19<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[2m2025-11-19T19:54:07.157134Z[0m [[32m[1minfo     [0m] [1mAgent `aisi-basic-agent-my-o3` finished running for `sample-specific-masks_6227908e-20d5-4a28-9fa6-73174810fbab.0`![0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m
[1763582047.1613078] PROBE: ExternalPythonCodingSolver.run(): self._run_agent() finished. Now running task.grade().
[1763582047.1621702] PROBE: --- paperbench/agents/utils.py: prepare_agent_dir_config() started. ---

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:20<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:21<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[2m2025-11-19T19:54:08.685022Z[0m [[32m[1minfo     [0m] [1mGrading the submission for sample-specific-masks_6227908e-20d5-4a28-9fa6-73174810fbab.0...[0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:22<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:23<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:24<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:25<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:26<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:27<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:28<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:29<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:30<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:31<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:32<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:33<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:34<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:35<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:36<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:37<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:38<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:39<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:40<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:41<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:42<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:43<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:44<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:45<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:46<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:47<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:48<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:49<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:50<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:51<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:52<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:53<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:54<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:55<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:56<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:57<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:58<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [12:59<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:00<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:01<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:02<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:03<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:04<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:05<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:06<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:07<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:08<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:09<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:10<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:11<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:12<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:13<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:14<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:15<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:16<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:17<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:18<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:19<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:20<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:21<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:22<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:23<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:24<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[2m2025-11-19T19:55:11.991678Z[0m [[32m[1minfo     [0m] [1mGrading for sample-specific-masks_6227908e-20d5-4a28-9fa6-73174810fbab.0 finished![0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m
[2m2025-11-19T19:55:12.032319Z[0m [[32m[1minfo     [0m] [1m[1;35mGrades saved to /disk/disk_20T/luoyujie/preparedness/project/paperbench/runs/2025-11-19T19-41-46-GMT_run-group_aisi-basic-agent-my-o3/sample-specific-masks_6227908e-20d5-4a28-9fa6-73174810fbab/pb_result.json[0m[0m [36mcomponent[0m=[35mpaperbench.nano.eval[0m

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval):  50%|█████     | 1/2 [13:25<00:01,  1.01s/it, corr=0, errs=0, fail=1]
PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[1763582113.5140676] PROBE: ExternalPythonCodingSolver._start_computer() finished.

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval): 100%|██████████| 2/2 [13:26<00:00, 251.65s/it, corr=0, errs=0, fail=1]asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval): 100%|██████████| 2/2 [13:26<00:00, 251.65s/it, corr=0, errs=0, fail=2][2m2025-11-19T19:55:13.871674Z[0m [[32m[1minfo     [0m] [1mGot back all results!         [0m [36mcomponent[0m=[35mnanoeval.evaluation[0m
[2m2025-11-19T19:55:13.874089Z[0m [[33m[1mwarning  [0m] [1m/disk/disk_20T/luoyujie/anaconda3/envs/paperbenchnew/lib/python3.11/site-packages/numpy/_core/_methods.py:222: RuntimeWarning: Degrees of freedom <= 0 for slice
  ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
[0m [36mcomponent[0m=[35mpy.warnings[0m
[2m2025-11-19T19:55:13.874801Z[0m [[33m[1mwarning  [0m] [1m/disk/disk_20T/luoyujie/anaconda3/envs/paperbenchnew/lib/python3.11/site-packages/numpy/_core/_methods.py:214: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
[0m [36mcomponent[0m=[35mpy.warnings[0m
[2m2025-11-19T19:55:13.876721Z[0m [[33m[1mwarning  [0m] [1m/disk/disk_20T/luoyujie/anaconda3/envs/paperbenchnew/lib/python3.11/site-packages/numpy/_core/fromnumeric.py:3860: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
[0m [36mcomponent[0m=[35mpy.warnings[0m
[2m2025-11-19T19:55:13.877065Z[0m [[33m[1mwarning  [0m] [1m/disk/disk_20T/luoyujie/anaconda3/envs/paperbenchnew/lib/python3.11/site-packages/numpy/_core/_methods.py:144: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
[0m [36mcomponent[0m=[35mpy.warnings[0m

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[2m2025-11-19T19:55:13.880356Z[0m [[32m[1minfo     [0m] [1mSummary:
{'metrics': {'mean_score': 0.281977847255625,
             'mean_score_by_paper': {'bridging-data-gaps': {'mean': 0.1392857142857143,
                                                            'n_runs': 1,
                                                            'run_1': 0.1392857142857143,
                                                            'std_err': nan},
                                     'sample-specific-masks': {'mean': 0.42466998022553576,
                                                               'n_runs': 1,
                                                               'run_1': 0.42466998022553576,
                                                               'std_err': nan}},
             'n_complete_tries': 1,
             'std_err': nan},
 'other_stats': {'n_is_valid_git_repo': 0,
                 'n_nontrivial_git_log': 0,
                 'n_repro_script_exists': 0,
                 'repro_mean_time': np.float64(nan)},
 'params': {'agent': 'aisi-basic-agent-my-o3',
            'code_only': True,
            'n_samples': 2,
            'n_tries': 1,
            'paper_split': 'dev',
            'skip_reproduction': False},
 'run_group_id': '2025-11-19T19-41-46-GMT_run-group_aisi-basic-agent-my-o3',
 'run_health': {'n_gradings_failed': 0,
                'n_reproductions_failed': 2,
                'n_rollouts_failed': 0}}
Evaluated 2 tasks with PaperBench-aisi-basic-agent-my-o3[0m [36mcomponent[0m=[35mnanoeval.evaluation[0m
[2m2025-11-19T19:55:13.880819Z[0m [[32m[1minfo     [0m] [1m[36m[1mEvalboard Summary: unknown, add in pr?[0m[0m [36mcomponent[0m=[35mnanoeval.evaluation[0m
asic-agent-my-o3 (r_id=251119194146BD7ZHZPI, m=nanoeval): 100%|██████████| 2/2 [13:26<00:00, 403.46s/it, corr=0, errs=0, fail=2]
[2m2025-11-19T19:55:13.881930Z[0m [[33m[1mwarning  [0m] [1mUnable to find model name, using fallback name='nanoeval'[0m [36mcomponent[0m=[35mnanoeval.eval[0m
[2m2025-11-19T19:55:13.883272Z[0m [[33m[1mwarning  [0m] [1mNo slack integration configured, so not sending user notification[0m [36mcomponent[0m=[35mnanoeval.library_config[0m
[1763582113.8853097] PROBE: --- await run(...) finished. ---
[1763582113.8853498] PROBE: --- async main() function finished. ---

PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]


PROBE_DB: Attempting to connect to self.database_file: [/tmp/nanoeval/251119194146SBM2TZM7.db]

[1763582113.9590068] PROBE: --- nanoeval_entrypoint(...) finished. ---
[1763582113.9590404] PROBE: --- End of script. ---
