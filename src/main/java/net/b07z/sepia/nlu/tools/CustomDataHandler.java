package net.b07z.sepia.nlu.tools;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Some useful stuff to handle files with e.g. training data in "exotic" custom format ^^.
 * 
 * @author Florian Quirin
 *
 */
public class CustomDataHandler {
	
	private static final String CUSTOM_DATA_SEPERATOR = " --- ";
	
	/**
	 * Import data from a custom "compact" file to a map with key=sentence, value=labels.
	 * @param filePath
	 * @return
	 * @throws IOException
	 */
	public static List<CompactDataEntry> importCompactData(String filePath) throws IOException{
		List<CompactDataEntry> trainData = new ArrayList<>();
		List<String> compactData = Files.readAllLines(Paths.get(filePath));
		int N = 0;
		for (String line : compactData){
			N++;
			if (line == null || line.isEmpty()){
				continue;
			}
			String[] keyVal = line.split(CUSTOM_DATA_SEPERATOR);
			if (keyVal.length > 3){
				throw new RuntimeException("Invalid format in line " + N + ": " + line);
			}
			int n = keyVal.length;
			String sentence = "";
			String labels = "";
			String intent = "";
			if (n > 0){
				sentence = keyVal[0].trim();
			}
			if (n > 1){
				labels = keyVal[1].trim();
			}
			if (n > 2){
				intent = keyVal[2].trim();
			}
			trainData.add(new CompactDataEntry(sentence, labels, intent));
		}
		return trainData;
	}

	/**
	 * Write data imported e.g. via {@link MalletDataHandler#importCompactTrainData} to file.
	 * @param filePath
	 * @param data
	 * @throws IOException
	 */
	public static void writeTrainData(String filePath, List<String> data) throws IOException{
		Files.write(Paths.get(filePath), data, StandardCharsets.UTF_8);
	}
	
	/**
	 * Write data imported e.g. via {@link MalletDataHandler#importCompactTrainData} to file, optionally without labels.
	 * @param filePath
	 * @param data
	 * @throws IOException
	 */
	public static void writeTestData(String filePath, List<String> data, boolean removeLabels) throws IOException{
		if (removeLabels){
			data.stream().forEach(line -> {
				line = line.replaceFirst("\\s.*", "").trim();
			});
		}
		Files.write(Paths.get(filePath), data, StandardCharsets.UTF_8);
	}
	
	/**
	 * Save settings to properties file.
	 * @param config_file - path and file
	 * @param config - Properties with settings to store
	 */
	public static void saveProperties(String config_file, Properties config) throws Exception{
		OutputStream out =null;
		File f = new File(config_file);
	    out = new FileOutputStream( f );
	    config.store(out, null);
	    out.flush();
	    out.close();
	}

	/**
	 * Load settings from properties file and return Properties.
	 * @param config_file - path and file
	 */
	public static Properties loadProperties(String config_file) throws Exception{
		BufferedInputStream stream=null;
		Properties config = new Properties();
		stream = new BufferedInputStream(new FileInputStream(config_file));
		config.load(stream);
		stream.close();
		return config;
	}

}
