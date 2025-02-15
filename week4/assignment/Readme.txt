Subtask A: Purpose to train pre-trained model. https://huggingface.co/docs/transformers/training
Step1: system setup: 
  1.1 end purpose to detect nonsense sentence base one 2 sentence which in 2 pandas columns. 
  1.2 Prepare folder: check if raw input file not creaated yet. os.path.exists()
  1.3 not --> clone to get raw data from git: !git clone
  1.4: set randome choose:
          from transformer import enable_full_determinism
          enable_full_determinism(42)

  1.5: set up varibale: batch_size, epoch, output_dir,...

Step2: Loading pretrain model. 
  2.1: loading model: 
          from transformer import AutoModelSequenceForClassification
          AutoModelSequenceForClassification.from_pretrained(model_name)
  2.2 Loading tokenization from this model. 
          from transformer import AutoTokenizer
          AutoTokenizer.from_pretrained(model_name)

Step3: Data preprocessing. 
  3.1 Create pandas data frame: id, correct_sent0; failse_sent1; label (0 or 1). 
          df = pandas.Dataframe()
  3.2 transfomr pandas data frome to Apache_arrow_table. (same pandas). 
          from datasets import Dataset
          train_dataset = Dataset.from_pandas(df)
          test_dataset = Dataser.from_pandas(dff)
