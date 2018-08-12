package net.b07z.sepia.nlu.trainers;

import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;

/**
 * Train OpenNlp to classify intents.
 * 
 * @author Florian Quirin
 *
 */
public class OpenNlpIntentTrainer implements IntentTrainer {
	
	private String modelFileBase;
	private String trainingFileBase;
	private String langCode;
	private DoccatModel model;
	
	public OpenNlpIntentTrainer(String propertiesFile, String trainDataFileBase, String modelOutputFileBase, String langCode) {
		this.trainingFileBase = trainDataFileBase;
		this.modelFileBase = modelOutputFileBase;
		this.langCode = langCode;
		//TODO: add properties
	}

	@Override
	public void train() {
		String trainingFile = this.trainingFileBase + "_" + langCode;
		InputStreamFactory isf = new InputStreamFactory() {
            public InputStream createInputStream() throws IOException {
                return new FileInputStream(trainingFile);
            }
        };

		Charset charset = Charset.forName("UTF-8");
        try (ObjectStream<String> lineStream = new PlainTextByLineStream(isf, charset);
        		ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);){
			
        	TrainingParameters trainingParams = TrainingParameters.defaultParams();
        	//TrainingParameters trainingParams = new TrainingParameters();
            //trainingParams.put(TrainingParameters.ITERATIONS_PARAM, 10);
            //trainingParams.put(TrainingParameters.CUTOFF_PARAM, 0);
        	
        	DoccatFactory docClassifierFactory = new DoccatFactory();
            this.model = DocumentCategorizerME.train(this.langCode, sampleStream, 
            		trainingParams, docClassifierFactory);
        
        }catch (Exception e){
        	System.err.println("Failed to train model: " + e.getMessage());
        	e.printStackTrace();
        }		
        
        if (this.model != null){
        	String modelFile = this.modelFileBase + "_" + langCode;
	        try (BufferedOutputStream modelOut = new BufferedOutputStream(new FileOutputStream(modelFile))){
	        	model.serialize(modelOut);
	        }catch (IOException e){
	        	System.err.println("Failed to store model: " + e.getMessage());
				e.printStackTrace();
			}
        }
	}

}
