package de.unimarburg.generateddatasetdetector;

import de.unimarburg.generateddatasetdetector.data.Database;
import de.unimarburg.generateddatasetdetector.data.Table;
import de.unimarburg.generateddatasetdetector.data.metadata.dependency.FunctionalDependency;
import de.unimarburg.generateddatasetdetector.data.metadata.dependency.InclusionDependency;
import de.unimarburg.generateddatasetdetector.data.metadata.dependency.MetanomeImpl;
import de.unimarburg.generateddatasetdetector.data.metadata.dependency.UniqueColumnCombination;
import de.unimarburg.generateddatasetdetector.utils.InputReader;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.io.File;
import java.util.Comparator;
import java.util.List;

import org.json.JSONException;
import org.json.JSONObject;

public class MetanomeMain {

    public void run(String inputPath, String outputFile) {
        try {
            File inputFile = new File(inputPath);
    
            if (!inputFile.exists()) {
                System.err.println("❌ Error: Input path not found -> " + inputPath);
                return;
            }
    
            // Detect dataset type (fakeData or realData)
            String dataType = detectDataType(inputFile);
            if (dataType == null) {
                System.err.println("❌ Error: Input must be inside TrainingData/fakeData or TrainingData/realData.");
                return;
            }
    
            if (inputFile.isDirectory()) {
                processDirectory(inputFile, dataType, outputFile);
            } else {
                processSingleFile(inputFile, dataType, outputFile);
            }
    
        } catch (Exception e) {
            System.err.println("❌ Error processing Metanome:");
            e.printStackTrace();
        }
    }
    
    private void processDirectory(File directory, String dataType, String outputFile) {
        System.out.println("📂 Processing directory: " + directory.getName());
    
        File outputDir = new File(directory.getParent(), "metanomeResults/" + directory.getName());
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
    
        for (File csvFile : directory.listFiles()) {
            if (csvFile.isFile() && csvFile.getName().endsWith(".csv")) {
                String outputFileName = csvFile.getName().replace(".csv", "_Results.json");
                String finalOutputFile = (outputFile != null && !outputFile.isEmpty())
                        ? outputFile
                        : new File(outputDir, outputFileName).getAbsolutePath();
                processTable(csvFile, finalOutputFile);
            }
        }
    }
    
    private void processSingleFile(File file, String dataType, String outputFile) {
        System.out.println("📄 Processing single file: " + file.getName());
    
        // Ensure the results are stored under TrainingData/{dataType}/metanomeResults/
        File parentDir = file.getParentFile(); // e.g., TrainingData/fakeData/
        File outputDir = new File(parentDir, "metanomeResults");
    
        if (!outputDir.exists()) {
            outputDir.mkdirs(); // Ensure directory exists
        }
    
        // Construct the output file path: TrainingData/{dataType}/metanomeResults/{filename}_Results.json
        String outputFileName = file.getName().replace(".csv", "_Results.json");
        File outputFilePath = new File(outputDir, outputFileName);
    
        // Use provided outputFile if given, otherwise use dynamically determined path
        String finalOutputFile = (outputFile != null && !outputFile.isEmpty())
                ? outputFile
                : outputFilePath.getAbsolutePath();
    
        processTable(file, finalOutputFile);
    }
    
    private void processTable(File file, String outputFile) {
        try {
            Table table = InputReader.readDataFile(file.getAbsolutePath());
            System.out.println("🔍 Analyzing table: " + table.getName());
    
            // Functional Dependency (FD) Search
            List<FunctionalDependency> fdResults = MetanomeImpl.executeFD(List.of(table));
            System.out.println("✅ FD Search completed. Results found: " + fdResults.size());
    
            // Unique Column Combination (UCC) Search
            List<UniqueColumnCombination> uccResults = MetanomeImpl.executeUCC(List.of(table));
            System.out.println("✅ UCC Search completed. Results found: " + uccResults.size());
    
            // Inclusion Dependency (IND) Search
            List<InclusionDependency> indResults = MetanomeImpl.executeIND(List.of(table));
            System.out.println("✅ IND Search completed. Results found: " + indResults.size());
    
            // Create JSON object to store results
            JSONObject tableJson = new JSONObject();
            tableJson.put("Table", table.getName());
            tableJson.put("NumberOfColumns", table.getColumns().size());
            tableJson.put("FDs_count", fdResults.size());
            tableJson.put("UCCs_count", uccResults.size());
            tableJson.put("INDs_count", indResults.size());
            tableJson.put("Max_FD_Length", fdResults.stream().map(FunctionalDependency::getLength).max(Integer::compareTo).orElse(0));
    
            // Write JSON results
            writeJsonToFile(tableJson, outputFile);
    
        } catch (Exception e) {
            System.err.println("❌ Error processing table: " + file.getName());
            e.printStackTrace();
        }
    }
    
    private static void writeJsonToFile(JSONObject jsonObject, String filePath) {
        try (FileWriter file = new FileWriter(filePath)) {
            file.write(jsonObject.toString(4)); // Pretty-print JSON
            file.flush();
            System.out.println("✅ Results saved to: " + filePath);
        } catch (IOException e) {
            System.err.println("❌ Error writing JSON file: " + e.getMessage());
        }
    }
    
    private String detectDataType(File inputFile) {
        File parent = inputFile.getParentFile();
        while (parent != null) {
            if (parent.getName().equals("fakeData")) {
                return "fakeData";
            } else if (parent.getName().equals("realData")) {
                return "realData";
            }
            parent = parent.getParentFile();
        }
        return null;
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage:");
            System.out.println("  java -jar MetanomeMain.jar --input-file <csv_or_directory> [--output-file <json>]");
            return;
        }
    
        String inputFile = null;
        String outputFile = null;
    
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--input-file") && i + 1 < args.length) {
                inputFile = args[i + 1];
            } else if (args[i].equals("--output-file") && i + 1 < args.length) {
                outputFile = args[i + 1];
            }
        }
    
        if (inputFile == null) {
            System.err.println("❌ Error: Missing required argument --input-file.");
            return;
        }
    
        System.out.println("🚀 Running Metanome on: " + inputFile);
        if (outputFile != null) {
            System.out.println("📂 Output will be saved to: " + outputFile);
        } else {
            System.out.println("📂 Output file not specified, it will be generated automatically.");
        }
    
        new MetanomeMain().run(inputFile, outputFile);
    }
}
