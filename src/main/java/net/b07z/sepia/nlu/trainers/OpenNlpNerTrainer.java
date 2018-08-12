package net.b07z.sepia.nlu.trainers;

import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.Map;

import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.NameSampleDataStream;
import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;

public class OpenNlpNerTrainer implements NerTrainer {
	
	private String modelFileBase;
	private String trainingFileBase;
	private Map<String, TokenNameFinderModel> models;
	private String[] labelNames;
	private String langCode;
	
	/**
	 * Create a trainer for one label.
	 * @param propertiesFile - tbd
	 * @param trainDataFileBase - 
	 * @param modelOutputFileBase - 
	 * @param labelNames - 
	 * @param langCode - ISO language code to tag the classifier results (e.g. "en")
	 * @throws Exception
	 */
	public OpenNlpNerTrainer(String propertiesFile, String trainDataFileBase, String modelOutputFileBase, String[] labelNames, String langCode) throws Exception{
		this.modelFileBase = modelOutputFileBase;
		this.trainingFileBase = trainDataFileBase;
		this.labelNames = labelNames;
		this.langCode = langCode;
		//TODO: add properties
		this.models = new HashMap<>();
	}

	@Override
	public void train() {
		for (String label : this.labelNames){
			train(label, this.langCode);
		}
	}
	
	/**
	 * Train a single label.
	 */
	public void train(String label, String langCode) {
		String trainingFile = this.trainingFileBase + "_" + label + "_" + langCode;
		InputStreamFactory isf = new InputStreamFactory() {
            public InputStream createInputStream() throws IOException {
                return new FileInputStream(trainingFile);
            }
        };
        
        String LANG = langCode;
        String LABEL = label;
        TokenNameFinderModel model = null;

        Charset charset = Charset.forName("UTF-8");
        try (ObjectStream<String> lineStream = new PlainTextByLineStream(isf, charset);
        		ObjectStream<NameSample> sampleStream = new NameSampleDataStream(lineStream)){
        	
        	TokenNameFinderFactory nameFinderFactory = new TokenNameFinderFactory();
        	model = NameFinderME.train(LANG, LABEL, sampleStream, 
            			TrainingParameters.defaultParams(), nameFinderFactory);
            
        }catch (Exception e){
        	System.err.println("Failed to train model: " + e.getMessage());
        	e.printStackTrace();
        }
        if (model != null){
        	this.models.put(LABEL, model);
        	String modelFile = this.modelFileBase + "_" + label + "_" + langCode;
	        try (BufferedOutputStream modelOut = new BufferedOutputStream(new FileOutputStream(modelFile))){
	        	model.serialize(modelOut);
	        }catch (IOException e){
	        	System.err.println("Failed to store model: " + e.getMessage());
				e.printStackTrace();
			}
        }
	}

}
