package de.unimarburg.generateddatasetdetector.algorithms;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.commons.csv.CSVRecord;
import de.unimarburg.generateddatasetdetector.data.Column;
import de.unimarburg.generateddatasetdetector.data.Table;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class ZipfsLaw {

    static TextPreprocessor textPreprocessor = new TextPreprocessor();


    public static boolean isNumeric(String str) {
        try {
            Float.parseFloat(str);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private static Map<String, Integer> calculateWordFrequency(List<String> textColumn) {
        Map<String, Integer> wordFrequencies = new HashMap<>();
        for (String text : textColumn) {
            String[] words = text.split("\\s+");
            for (String word : words) {
                if (!word.isEmpty()) {
                    wordFrequencies.merge(word, 1, Integer::sum);
                }
            }
        }
        return wordFrequencies.entrySet().stream()
                .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (e1, e2) -> e1,
                        LinkedHashMap::new
                ));
    }

    private static void displayZipfsLaw(Map<String, Integer> wordFrequencies) {
        int rank = 1;
        System.out.printf("%-10s%-10s%-10s\n", "Rank", "Word", "Frequency");
        for (Map.Entry<String, Integer> entry : wordFrequencies.entrySet()) {
            System.out.printf("%-10d%-10s%-10d\n", rank++, entry.getKey(), entry.getValue());
        }
    }

    private static void plotZipfsLaw(String columnName, Map<String, Integer> wordFrequencies) {
        List<Integer> ranks = new ArrayList<>();
        List<Integer> frequencies = new ArrayList<>();
        List<Double> optimalFrequencies = new ArrayList<>();
        int totalFrequency = wordFrequencies.values().stream().mapToInt(Integer::intValue).sum();
        int rank = 1;

        for (int frequency : wordFrequencies.values()) {
            ranks.add(rank++);
            frequencies.add(frequency);
        }
        for (int r = 1; r <= ranks.size(); r++) {
            optimalFrequencies.add((totalFrequency / harmonicSum(ranks.size())) / r);
        }


        Plot plt = Plot.create();
        plt.plot()
                .add(ranks, frequencies)
                .label("Zipf's Law")
                .linestyle("-")
                .color("blue");
        plt.plot()
                .add(ranks, optimalFrequencies)
                .label("Optimal Zipf Distribution")
                .linestyle("--")
                .color("pink");
        plt.title("Zipf's Law for Column: " + columnName);
        plt.xlabel("Rank");
        plt.ylabel("Frequency");
        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            System.err.println("Error displaying plot: " + e.getMessage());
        }
    }

    private static double harmonicSum(int n) {
        double sum = 0.0;
        for (int i = 1; i <= n; i++) {
            sum += 1.0 / i;
        }
        return sum;
    }


    public static Map<String, Double> calcZipfWithChiSquareFromTable(List<Table> tables) {
        Map<String, Double> chiSquareResults = new HashMap<>();
        Map<String, List<String>> textColumns = new HashMap<>();
        for (Table table : tables) {
            for (Column column : table.getColumns()) {
                textColumns.put(column.getLabel(), new ArrayList<>());
                List<String> textValues = new ArrayList<>();
                for (String value : column.getValues()) {
                    String processedValue = textPreprocessor.process(value.trim());
                    if (!isNumeric(processedValue)) {
                        textColumns.get(column.getLabel()).add(processedValue);
                    }
                }
                if (!textValues.isEmpty()) {
                    Map<String, Integer> wordFrequencies = calculateWordFrequency(textValues);
                    double chiSquare = DatasetAuthenticityAnalyzer.analyzeZipfsLaw(wordFrequencies);
                    chiSquareResults.put(column.getLabel(), chiSquare);
                }
            }
        }
            Map<String, Integer> wordFrequencies = new HashMap<>();
            Scanner zipfScanner = new Scanner(System.in);
            for (Table table : tables) {
                for (Column column : table.getColumns()) {
                    if (!textColumns.get(column.getLabel()).isEmpty()) {
                        System.out.println("Zipf's Law for column: " + column);
                        wordFrequencies = calculateWordFrequency(textColumns.get(column.getLabel()));
                        //Remove words that only appear once, twice or three times
                        wordFrequencies.entrySet().removeIf(entry -> entry.getValue() < 4);
                        displayZipfsLaw(wordFrequencies);
                        System.out.println("Do you want to plot the Zipf distribution for column: " + column + "? (y/n)");
                        String answer = zipfScanner.nextLine();
                        if (answer.equals("y")) {
                            System.out.println("Plot for column: " + column.getLabel());
                            plotZipfsLaw(column.getLabel(), wordFrequencies);
                        }
                    }
                }
            }
            return chiSquareResults;
        }
}


