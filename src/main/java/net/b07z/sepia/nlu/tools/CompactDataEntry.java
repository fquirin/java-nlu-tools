package net.b07z.sepia.nlu.tools;

/**
 * Class that holds an entry imported from a compact data source.
 * 
 * @author Florian Quirin
 *
 */
public class CompactDataEntry {
	
	String sentence;
	String labels;
	String intent;
	
	public CompactDataEntry(String sentence, String labels, String intent){
		this.sentence = sentence;
		if (labels != null && !labels.isEmpty() && !labels.equals("#")) this.labels = labels;
		if (intent != null && !intent.isEmpty() && !intent.equals("#")) this.intent = intent;
	}
	
	public String getSentence(){
		return sentence;
	}
	
	public String getLabels(){
		return labels;
	}
	
	public String getIntent(){
		return intent;
	}

}
