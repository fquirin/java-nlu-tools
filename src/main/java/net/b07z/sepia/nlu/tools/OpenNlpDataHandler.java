package net.b07z.sepia.nlu.tools;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import net.b07z.sepia.nlu.tokenizers.Tokenizer;

/**
 * Writes data in different formats to a file that can be used to train OpenNLP NER.
 * 
 * @author Florian Quirin
 *
 */
public class OpenNlpDataHandler implements CompactDataHandler{
	
	@Override
	public List<String> importTrainDataNer(Collection<CompactDataEntry> compactData, Tokenizer tokenizer, boolean failOnMismatch, String filterLabel){
		List<String> trainDataLines = new ArrayList<>();
		String defaultLabel = tokenizer.getDefaultLabel();
		int n = 0;
		for (CompactDataEntry cde : compactData){
			n++;
			String sentence = cde.getSentence();
			String labelString = cde.getLabels();
			if (labelString == null){
				if (failOnMismatch){
					throw new RuntimeException("Skipped sentence - Missing labels in line " + n + ": " + sentence);
				}else{
					System.err.println("Skipped sentence - Missing labels in line " + n + ": " + sentence);
				}
				continue;
			}
			labelString = tokenizer.getBeginningOfSentenceLabel() + labelString + tokenizer.getEndOfSentenceLabel();
			List<String> tokens = tokenizer.getTokens(sentence);
			String[] labels = labelString.trim().split("\\s+");
			boolean gotError = false;
			if (tokens.size() != labels.length){
				if (failOnMismatch){
					throw new RuntimeException("Tokens and labels size does not match in line " + n + ": " + sentence);
				}else{
					System.err.println("Skipped sentence - Tokens and labels size does not match in line " + n + ": " + sentence);
				}
				gotError = true;
			}
			if (!gotError){
				StringBuffer newLine = new StringBuffer();
				for (int i=0; i<tokens.size(); i++){
					if ((filterLabel != null && !filterLabel.isEmpty() && !labels[i].equals(filterLabel)) || labels[i].equals(defaultLabel)){
						newLine.append(" ")
							.append(tokens.get(i))
							.append(" ");
					}else{
						newLine.append(" <START:").append(labels[i]).append("> ")
							.append(tokens.get(i))
							.append(" <END> ");
					}
				}
				trainDataLines.add(newLine.toString());
			}
		}
		return trainDataLines;
	}

	@Override
	public List<String> importTrainDataIntent(Collection<CompactDataEntry> compactData, Tokenizer tokenizer, String filterIntent) {
		List<String> trainDataLines = new ArrayList<>();
		int n = 0;
		for (CompactDataEntry cde : compactData){
			n++;
			if (cde.intent == null){
				System.err.println("Skipped sentence - Missing intent in line " + n + ": " + cde.sentence);
				continue;
			}
			if (filterIntent != null && !filterIntent.isEmpty() && !cde.intent.equals(filterIntent)){
				continue;
			}else{
				trainDataLines.add(cde.intent + " " + tokenizer.normalizeSentence(cde.sentence));
			}
		}
		return trainDataLines;
	}
}
