package net.b07z.sepia.nlu.classifiers;

import java.util.Collection;
import java.util.Comparator;
import java.util.Set;
import java.util.TreeSet;

/**
 * Hold result of intent classifiers.
 * 
 * @author Florian Quirin
 *
 */
public class IntentEntry {
	private String sentence;
	private String intent;
	private double certainty = -1.0;
	
	public IntentEntry(String sentence, String intent, double certainty){
		this.sentence = sentence;
		this.intent = intent;
		this.certainty = certainty;
	}
	
	public String getSentence(){
		return this.sentence;
	}
	
	public String getIntent(){
		return this.intent;
	}
	
	public double getCertainty(){
		return this.certainty;
	}
	
	@Override
	public String toString(){
		return intent;
	}
	
	public String toStringComplex(){
		return (intent + " (" + certainty + ")");
	}
	
	//--------------------------------
	
	/**
	 * Make a collection (usually a set with special comparator) for intent entries.
	 */
	public static Collection<IntentEntry> makeIntentCollection(){
		Comparator<IntentEntry> comp = (IntentEntry ie1, IntentEntry ie2) -> { 
			int dc = Double.compare(ie2.getCertainty(), ie1.getCertainty());
			if (dc == 0){
				int sc = ie1.getIntent().compareTo(ie2.getIntent());
				if (sc == 0) return 0;	//if intent has same name and certainty it is removed from set
				else return 1;			//keep input order in this case
			}else{
				return dc;
			}
		};
		Set<IntentEntry> intentCollection = new TreeSet<>(comp);
		return intentCollection;
	}
	/**
	 * Get best intent of a collection of intent entries (usually just the first entry).
	 * @param entries - entries created with the collection given in this class
	 * @return
	 */
	public static IntentEntry getBestIntent(Collection<IntentEntry> entries){
		return entries.iterator().next();
	}
}
