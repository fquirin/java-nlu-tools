# Java NLU Tools
Java tools to do natural language processing like NER and intent classification on short sentences.  
These tools provide an abstraction layer to [MALLET: A Machine Learning for Language Toolkit](http://mallet.cs.umass.edu/) (Common Public License) and [Apache OpenNLP](https://opennlp.apache.org/) (Apache License Version 2.0). Stanford CoreNLP is supported as well but due to their GPL license it is implemented as [plugin](https://github.com/fquirin/java-nlu-tools-corenlp).  

# How to
All training data can be stored in a common, easy to read file with the following format:  
```
How is the weather? --- O O O O --- WEATHER
How's the weather in Berlin? --- O O O O LOCATION --- WEATHER
I need a taxi --- O O O O --- TAXI_SERVICE
I need a cap to go to Pier 39 --- O O O O O O O LOCATION LOCATION --- TAXI_SERVICE
...
```
Each line starts with a sentence in raw format followed by labels that are used to extract named entities and an intent connected to the sentence. Each part is separated by 3 dashes with space " --- ". The default label for unnamed words (tokens) is "O", labels can be chosen as you like. The more sentences you have the better, but 15 per intent and named-entity should be fine for testing.  
  
To start training your model first choose your toolkit and extract the data to the required format:
```
//Import training data from compact custom format
Collection<CompactDataEntry> trainData = CustomDataHandler.importCompactData(compactTrainDataFile);

//Declare a tokenizer for our model
Tokenizer tokenizer = new RealLifeChatTokenizer("", "", "");

//Store train data for OpenNLP intent classification
CompactDataHandler cdh = new OpenNlpDataHandler();
List<String> trainDataLines = cdh.importTrainDataIntent(trainData, tokenizer, null);
CustomDataHandler.writeTrainData(trainFile, trainDataLines);
```
Then you can start training:
```
//Start training with intent trainer
IntentTrainer trainer = new OpenNlpIntentTrainer(null, trainFileBase, modelFileBase, languageCode);
trainer.train();
```
To test your model load it in a new classifier and call it like this:
```
IntentClassifier ic = new OpenNlpIntentClassifier(modelFileBase, tokenizer, languageCode);
Collection<IntentEntry> intentResults = ic.analyzeSentence(sentence);
IntentEntry bestResult = IntentEntry.getBestIntent(intentResults);
String bestIntent = bestResult.getIntent();
double bestCertainty = bestResult.getCertainty();
```
Check out the [examples](https://github.com/fquirin/java-nlu-tools/tree/master/src/main/java/net/b07z/sepia/nlu/examples) for each toolkit to get a complete overview of the export-train-test procedure.  
