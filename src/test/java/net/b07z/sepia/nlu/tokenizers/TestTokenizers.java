package net.b07z.sepia.nlu.tokenizers;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

public class TestTokenizers {

	@Test
	public void testRealLifeChatTokenizer() {
		
		List<String> sentences = Arrays.asList(
				"Hey what's up????",
				"A text {A=1,t=2} with some special [spec.] characters!?!!?!??",
				"Put milk, water and bread on my to-do list",
				"I wanna fly to LA"
		);
		List<String> expected = Arrays.asList(
				"[BOS, hey, whats, up, EOS]",
				"[BOS, a, text, a=1, SEP, t=2, with, some, special, spec., characters, EOS]",
				"[BOS, put, milk, SEP, water, and, bread, on, my, to-do, list, EOS]",
				"[BOS, i, wanna, fly, to, la, EOS]"
		);	
		
		Tokenizer tokenizer = new RealLifeChatTokenizer();
		
		for (int i=0; i<sentences.size(); i++){
			String s = sentences.get(i);
			List<String> tokens = tokenizer.getTokens(s);
			//System.out.println(s);
			//System.out.println(tokens.toString());
			assertTrue(tokens.toString().equals(expected.get(i)));
		}
		
		assertTrue(true);
	}

}
