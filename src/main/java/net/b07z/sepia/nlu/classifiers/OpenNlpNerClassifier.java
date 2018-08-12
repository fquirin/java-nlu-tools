package net.b07z.sepia.nlu.classifiers;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import net.b07z.sepia.nlu.tokenizers.RealLifeChatTokenizer;
import net.b07z.sepia.nlu.tokenizers.Tokenizer;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.util.Span;

/**
 * OpenNlp implementation of NER classifier.
 * 
 * @author Florian Quirin
 *
 */
public class OpenNlpNerClassifier implements NerClassifier{
		
	private Tokenizer tokenizer;
	private Map<String, TokenNameFinderModel> models;
	private String[] labels;
	private String languageCode;
	
	/**
	 * Create NER classifier with model and tokenizer.
	 * @param modelFileBase
	 * @param tokenizer
	 * @param labels
	 * @param languageCode
	 * @throws Exception
	 */
	public OpenNlpNerClassifier(String modelFileBase, Tokenizer tokenizer, String[] labels, String languageCode) throws Exception {
		//load model
		this.models = new HashMap<>();
		this.languageCode = languageCode;
		for (String label : labels){
			String modelFile = modelFileBase + "_" + label + "_" + this.languageCode;
			try (InputStream modelIn = new FileInputStream(modelFile)){
				this.models.put(label, new TokenNameFinderModel(modelIn));
			}
		}
		this.tokenizer = tokenizer;
		this.labels = labels;
	}

	@Override
	public List<NerEntry> analyzeSentence(String sentence) {
		//Normalize and get tokens
		List<String> tokens = tokenizer.getTokens(sentence);
		TokenModelLabels[] multiModelLabels = new TokenModelLabels[tokens.size()];
		String defaultLabel = tokenizer.getDefaultLabel();
		
		//Find entities for all labels - result will be in multiModelLabels
		findLabelsForAllModels(tokens, multiModelLabels);
		
		//Store
		List<NerEntry> nerEntries = new ArrayList<>();
		for (int i=0; i<tokens.size(); i++){
			String t = tokens.get(i);
			if (multiModelLabels[i] != null){
				String bestLabel = multiModelLabels[i].getBestLabel();
				t = multiModelLabels[i].token; 		//just to be sure
				NerEntry ne = new NerEntry(getOriginalToken(t, sentence), t, bestLabel, multiModelLabels[i].getAllLabels());
				ne.setCertainty(multiModelLabels[i].getBestCertainty());
				nerEntries.add(ne);
			}else{
				String bestLabel = defaultLabel;
				TokenLabel tl = new TokenLabel(t, bestLabel, -1.0);
				NerEntry ne = new NerEntry(getOriginalToken(t, sentence), t, bestLabel, Arrays.asList(tl));
				nerEntries.add(ne);
			}
		}
		return nerEntries;
	}
	
	/**
	 * Run through all models and search labels 
	 * @param tokens
	 * @param multiModelLabels
	 */
	public void findLabelsForAllModels(List<String> tokens, TokenModelLabels[] multiModelLabels){
		//Find entities for all labels
		for (String label : this.labels){
			NameFinderME nerFinder;
			try{
				nerFinder = new NameFinderME(this.models.get(label));
			}catch (Exception e){
				throw new RuntimeException("Could not get model: " + e.getMessage());
			}
			Span[] nerSpans = nerFinder.find(tokens.toArray(new String[tokens.size()]));
			boolean hasSpans = (nerSpans.length > 0);
			if (hasSpans){
				for (Span s : nerSpans){
					int begin = s.getStart();
					int end = s.getEnd();
					if ((end-begin) != 1){
						throw new RuntimeException("Error in OpenNLP labels! We expect span size 1!");
					}
					if (multiModelLabels[begin] == null){
						multiModelLabels[begin] = new TokenModelLabels(tokens.get(begin));
					}
					multiModelLabels[begin].addLabel(label, s.getProb());
				}
			}
		}
	}
	
	@Override
	public List<NerEntry> getEntities(String sentence, boolean fuseSame, boolean removeDefaultLabel) {
		//Normalize and get tokens
		List<String> tokens = tokenizer.getTokens(sentence);
		TokenModelLabels[] multiModelLabels = new TokenModelLabels[tokens.size()];
		String defaultLabel = tokenizer.getDefaultLabel();
		
		//Find entities for all labels - result will be in multiModelLabels
		findLabelsForAllModels(tokens, multiModelLabels);
		
		//filter special labels
		String commonlabelRegex = RealLifeChatTokenizer.getCommonLabelRegEx(tokenizer, removeDefaultLabel);
		
		//Store
		List<NerEntry> nerEntries = new ArrayList<>();
		NerEntry ne = null;
		String lastLabel = "";
		
		for (int i=0; i<tokens.size(); i++){
			String token = tokens.get(i);
			String label = defaultLabel;
			double certainty = -1.0;
			Set<TokenLabel> allLabels = null;
			if (multiModelLabels[i] != null){
				label = multiModelLabels[i].getBestLabel();
				allLabels = multiModelLabels[i].getAllLabels();
				certainty = multiModelLabels[i].getBestCertainty();
			}
			//add to previous entry?
			if (fuseSame && ne != null && label.equals(lastLabel)){
				ne.addToken(token);
				ne.addOriginalToken(getOriginalToken(token, sentence));
				//NOTE: if you fuse two labels and they had different "alternatives" this might make certainties unreliable 
				ne.addAllLabels(allLabels);
			
			//make new entry
			}else{
				//skip all labels that are common
				if (!label.matches(commonlabelRegex)){
					lastLabel = label;
					ne = new NerEntry(getOriginalToken(token, sentence), token, label, allLabels);
					ne.setCertainty(certainty);
					nerEntries.add(ne);
				}else{
					ne = null;
				}
			}
		}
		return nerEntries;
	}
	
	//This is a kind of "heavy" way to get back the original token before it was converted to lower-case ... :see_no_evil:
	private String getOriginalToken(String normToken, String orgSentence){
		Matcher matcher = Pattern.compile("(?i)" + normToken).matcher(orgSentence);
		if (matcher.find()){
			return matcher.group();
		}else{
			return normToken;
		}
	}

}
