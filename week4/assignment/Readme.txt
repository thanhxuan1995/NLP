Subtask A: Purpose to train pre-trained model. https://huggingface.co/docs/transformers/training
Step1 (system set up) --> step2 (loading pretrain model) --> step3 (data preprocessing) --> step4 (tokenize data) --> step5 (fine tune model) --> step6 (make prediction).

Step1: system setup: 
  1.1 end purpose to detect nonsense sentence base one 2 sentence pair input which in 2 pandas columns. 
  1.2 Prepare folder: check if raw input file not creaated yet. os.path.exists()
  1.3 not --> clone to get raw data from git: !git clone
  1.4: set randome choose:
          from transformer import enable_full_determinism
          enable_full_determinism(42)

  1.5: set up varibale: batch_size, epoch, output_dir,...

Step2: Loading pretrain model. 
  2.1: loading model: 
          from transformer import AutoModelSequenceForClassification
          AutoModelSequenceForClassification.from_pretrained(model_name) --> model
  2.2 Loading tokenization from this model. 
          from transformer import AutoTokenizer
          AutoTokenizer.from_pretrained(model_name) --> tokenizer

Step3: Data preprocessing. 
  3.1 Create pandas data frame: id, correct_sent0; failse_sent1; label (0 or 1). 
          df = pandas.Dataframe()
  3.2 transfomr pandas data frome to Apache_arrow_table. (same pandas). 
          from datasets import Dataset
          train_dataset = Dataset.from_pandas(df)
          test_dataset = Dataser.from_pandas(dff) --> {'id': 2.
                                                        'sent0': "xu dep trai",
                                                        'sent1': "Xu xau trai",
                                                        'label': 1}
Step4: Tokenize dataset to transorm batch (columns 'sent0' and 'sent1' to tokenization
  4.1: using map function to perform each row of dataset as batch. 
          train_dataset = train_dataset.map(lambda x: func(x))
                x --> -> {'id': 2.
                          'sent0': "xu dep trai",
                          'sent1': "Xu xau trai",
                          'label': 1}
  4.2 define func to get sent0 and sent1 put into tokenization. 
          tokenizer(x['sent0'], x['sent1'], padding = True, Truncation = True, max_length)
           --> it make sure to return pair values: 
                  [CLS] I love natural language processing [SEP] It is a fascinating field [SEP]

  4.3 Result look like this: 
          {'id': 6252, 
            'sent0': 'a duck walks on three legs', 
            'sent1': 'a duck walks on two legs', 
            'label': 0, 
            '__index_level_0__': 6252, 
            'input_ids': [0, 102, 15223, 5792, 15, 130, 5856, 2, 2, 102, 15223, 5792, 15, 80, 5856,                             2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                               0, 0, 0]}
(0, '<s>'),
 (102, 'a'),
 (15223, 'Ġduck'),
 (5792, 'Ġwalks'),
 (15, 'Ġon'),
 (130, 'Ġthree'),
 (5856, 'Ġlegs'),
 (2, '</s>'),
 (2, '</s>'),
 (102, 'a'),
 (15223, 'Ġduck'),
 (5792, 'Ġwalks'),
 (15, 'Ġon'),
 (80, 'Ġtwo'),
 (5856, 'Ġlegs'),
 (2, '</s>')]

5. Fune tuning model. 
      5.1 Define training arguments to fine tune using TraningASrguments. 
            from transfomer import TrainingArguments
            arugment = TrainingArguments(output_dir, train_batch_size, learning_rate,                             save_strategy...)  
      5.2 put train model, train arguments, train_data, test_data into placeholder Trainer. 
            trainer = Trainer(
                            model,
                            args,
                            train_dataset,
                            eval_dataset
            )
      5.3 call trainer.train()

6. Make prediction and confusion matrix. 
      6.1 history = train.prteidction(test_dataset)
      6.2 test_data["prediction"] = history.predictions.argmax()
      6.3: import evaluate
            accuracy = evaluate.load("accuracy")
            final_acc = accuracy.compute(prediction, references).
            
