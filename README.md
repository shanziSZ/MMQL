Train or test the model:
  1. Install the requirements
  2. Run main.py for train or test

Error prompt prune:
  1. Result file without using prompt-QA during MQL prediction. The file is provided in "./saved_models/2024Jan26-121420_rebuttal_SLAKE_No_QA/estimate_results.pkl"
  2. Result file using prompt-QA during MQL prediction. In this anonymize repository, we provided "./saved_models/2024Jan30-125314_rebuttal_SLAKE_real/estimate_results.pkl". In this results, the answer in prompt-QA is predicted by other single-question-learning model(pesudo label). You can also change to other files.
  3. The program will generate a "flitered_results_*.txt".

Case accuracy:
  1. Use the result file, such as "flitered_results_*.txt".
  2. It generates the case accuracy.
