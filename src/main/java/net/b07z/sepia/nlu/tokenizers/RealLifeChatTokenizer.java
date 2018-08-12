package net.b07z.sepia.nlu.tokenizers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

/**
 * A tokenizer that tries to handle sloppy written real life text from short chats.
 * 
 * @author Florian Quirin
 *
 */
public class RealLifeChatTokenizer implements Tokenizer {
	
	private String BOS_TOKEN = "BOS";		//beginning of sentence
	private String EOS_TOKEN = "EOS";		//end of sentence
	private String SEPARATOR_TOKEN = "SEP";	//separator
	
	private String DEFAULT_LABEL = "O";		//default label
	
	/**
	 * Create tokenizer with default settings.
	 */
	public RealLifeChatTokenizer(){
		BOS_TOKEN = " " + BOS_TOKEN + " ";
		EOS_TOKEN = " " + EOS_TOKEN + " ";
		SEPARATOR_TOKEN = " " + SEPARATOR_TOKEN + " ";
	}
	/**
	 * Create tokenizer with custom BOS, EOS and SEP tokens.
	 */
	public RealLifeChatTokenizer(String BOS, String EOS, String SEP){
		BOS_TOKEN = " " + BOS + " ";
		EOS_TOKEN = " " + EOS + " ";
		SEPARATOR_TOKEN = " " + SEP + " ";
	}
	
	@Override
	public String getBeginningOfSentenceLabel() {
		return BOS_TOKEN;
	}
	@Override
	public String getEndOfSentenceLabel() {
		return EOS_TOKEN;
	}
	@Override
	public String getSeparatorLabel() {
		return SEPARATOR_TOKEN;
	}
	@Override
	public String getDefaultLabel() {
		return DEFAULT_LABEL;
	}

	@Override
	public List<String> getTokens(String sentence) {
		//normalize
		String normalizedSentence = sentence.toLowerCase();
		normalizedSentence = replaceEndOfLineChars(normalizedSentence);
		normalizedSentence = replaceSeparatorChars(normalizedSentence);
		normalizedSentence = removeSpecialChars(normalizedSentence);
		
		//add BOS tokens
		normalizedSentence = BOS_TOKEN + normalizedSentence;
		
		//split
		List<String> tokens = new ArrayList<String>(Arrays.asList(normalizedSentence.trim().split(""
				+ "\\n+|\\r+|\\t+|\\s+"
		+ "")));
		
		return tokens;
	}
	
	@Override
	public String normalizeSentence(String sentence) {
		return String.join(" ", getTokens(sentence));
	}
	
	/**
	 * Replace typical end-of-line characters with one common symbol: EOS_TOKEN.
	 * @param sentence - raw input text in lower-case!
	 * @return
	 */
	private String replaceEndOfLineChars(String sentence){
		String sClean =  sentence.replaceAll("(" +
				"!+|\\?+|;+|\\.(\\s+|$)"
			+ ")", EOS_TOKEN);
		sClean =  sClean.replaceAll("(" + Pattern.quote(EOS_TOKEN) + ")(\\1+)", EOS_TOKEN);
		if (!sClean.endsWith(EOS_TOKEN)){
			sClean += EOS_TOKEN;
		}
		return sClean.trim();
	}
	
	/**
	 * We don't expect them but take them if we have the chance ^^. 
	 * Replaces separators like "," with common symbol: SEP.
	 * @param sentence - raw input text in lower-case!
	 * @return
	 */
	private String replaceSeparatorChars(String sentence){
		return sentence.replaceAll("(" +
				",|\\|"
			+ ")", SEPARATOR_TOKEN).trim();
	}
	
	/**
	 * Remove special characters.
	 * @param sentence - raw input text
	 * @return
	 */
	private String removeSpecialChars(String sentence){
		String sClean = sentence.replaceAll("'(\\w)", "$1").trim(); 	//what's -> whats
		sClean = sClean.replaceAll("(" +
				"!|\\?|;|,|:|\\(|\\)|\"|'|\\[|\\]|\\{|\\}"
			+ ")", " ").trim();
		return sClean;
	}
	
	//----------------- helpers -------------------
	
	//Create a regular expression that can filter the common labels (usually not required in result).
	public static String getCommonLabelRegEx(Tokenizer tokenizer, boolean removeDefaultLabel){
		StringBuffer sb = new StringBuffer();
		if (removeDefaultLabel){
			sb.append(Pattern.quote(tokenizer.getDefaultLabel()));
		}
		String BOS = tokenizer.getBeginningOfSentenceLabel().trim();
		String EOS = tokenizer.getEndOfSentenceLabel().trim();
		String SEP = tokenizer.getSeparatorLabel().trim();
		if (!BOS.isEmpty()){
			sb.append("|");
			sb.append(Pattern.quote(BOS));
		}
		if (!EOS.isEmpty()){
			sb.append("|");
			sb.append(Pattern.quote(EOS));
		}
		if (!SEP.isEmpty()){
			sb.append("|");
			sb.append(Pattern.quote(SEP));
		}
		String commonlabelRegex = sb.toString().replaceFirst("^\\|", "").toString();
		//System.out.println("commonlabelRegex: " + commonlabelRegex); 			//DEBUG
		//String commonlabelRegex = "O" + "|" + BOS + "|" + EOS + "|" + SEP;
		return commonlabelRegex;
	}
}
