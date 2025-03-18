package de.unimarburg.generateddatasetdetector.algorithms;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import de.unimarburg.generateddatasetdetector.data.Column;
import de.unimarburg.generateddatasetdetector.data.Table;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.util.*;


public class Benford {

    //First implement a method that checks if the column of a is a number
    public static boolean isNumeric(String str) {
        try {
            Float.parseFloat(str);
            return true;
        } catch(NumberFormatException e){
            return false;
        }
    }

    //Returns the leading digit of a number that can also handle negative numbers and numbers with a comma
    public static int getLeadingDigit(float x) {
        x = Math.abs(x);
        while (x >= 10) {
            x /= 10;
        }
        return (int)x;

    }

    private static void printBenfordDistribution(Map<Integer, Integer> benfordCounts) {
        int total = benfordCounts.values().stream().mapToInt(Integer::intValue).sum();
        for (int i = 1; i <= 9; i++) {
            double frequency = (benfordCounts.get(i) * 100.0) / total;
            System.out.printf("Leading digit %d: %.2f%%\n", i, frequency);
        }
    }

    private static void plotBenfordDistribution(Map<Integer, Integer> benfordCounts) {
        List<Double> frequencies = new ArrayList<>();
        List<Integer> digits = new ArrayList<>();

        int total = benfordCounts.values().stream().mapToInt(Integer::intValue).sum();
        for (int i = 1; i <= 9; i++) {
            digits.add(i);
            double frequency = (benfordCounts.get(i) * 100.0) / total;
            frequencies.add(frequency);
        }

        Plot plt = Plot.create();
        plt.plot().add(digits, frequencies);
        //Add optimal Benford distribution
        List<Double> benford = new ArrayList<>();
        for (int i = 1; i <= 9; i++) {
            benford.add(Math.log10(1 + 1.0 / i) * 100);
        }
        plt.plot()
                .add(digits, benford)
                .label("Benford's Law")
                .linestyle("--");
        plt.xlabel("Leading Digit");
        plt.ylabel("Frequency (%)");
        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            System.err.println("Error displaying plot: " + e.getMessage());
        }
    }

    //Calculate leading digit for each column that is a number
    public static Map<String, Map<Integer, Integer>> calcBenford(Iterable<CSVRecord> data) throws PythonExecutionException, IOException {
        String[] headers = data.iterator().next().toMap().keySet().toArray(new String[0]);
        Map<String, Map<Integer, Integer>> benfordCounts = new HashMap<>();

        for (String column : headers) {
            benfordCounts.put(column, new HashMap<>());
            for (int i = 1; i <= 9; i++) {
                benfordCounts.get(column).put(i, 0); // Initialize counts to 0
            }
        }
        for (CSVRecord record : data) {
            for (String column : headers) {
                String value = record.get(column).trim();
                if (isNumeric(value)) {
                    int leadingDigit = getLeadingDigit(Float.parseFloat(value));
                    benfordCounts.get(column).merge(leadingDigit, 1, Integer::sum);
                }
            }
        }
        Scanner scanner = new Scanner(System.in);
        for (String column : headers) {
            if (benfordCounts.get(column).values().stream().mapToInt(Integer::intValue).sum() > 0) {
                System.out.println("Benford's Law for column: " + column);
                printBenfordDistribution(benfordCounts.get(column));
                System.out.println("Do you want to plot the Benford distribution for column: " + column + "? (y/n)");
                String answer = scanner.nextLine();
                if (answer.equals("y")) {
                    System.out.println("Plot for column: " + column);
                    plotBenfordDistribution(benfordCounts.get(column));
                }
            }
        }
        //scanner.close();
        return benfordCounts;
    }

    public static Map<String, Map<Integer, Integer>> calcBenfordFromTable(List<Table> tables){
            Map<String, Map<Integer, Integer>> benfordCounts = new HashMap<>();

            for (Table table : tables) {
                for (Column column : table.getColumns()) {
                    if (!benfordCounts.containsKey(column.getLabel())) {
                        benfordCounts.put(column.getLabel(), new HashMap<>());
                        for (int i = 1; i <= 9; i++) {
                            benfordCounts.get(column.getLabel()).put(i, 0); // Initialize counts to 0
                        }
                    }
                    for (String value : column.getValues()) {
                        if (isNumeric(value)) {
                            int leadingDigit = getLeadingDigit(Float.parseFloat(value));
                            benfordCounts.get(column.getLabel()).merge(leadingDigit, 1, Integer::sum);
                        }
                    }
                }
            }
        Scanner scanner = new Scanner(System.in);
        for (Table table : tables) {
            for (Column column : table.getColumns()) {
                if (benfordCounts.get(column.getLabel()).values().stream().mapToInt(Integer::intValue).sum() > 0) {
                    System.out.println("Benford's Law for column: " + column.getLabel());
                    printBenfordDistribution(benfordCounts.get(column.getLabel()));
                    System.out.println("Do you want to plot the Benford distribution for column: " + column.getLabel() + "? (y/n)");
                    String answer = scanner.nextLine();
                    if (answer.equals("y")) {
                        System.out.println("Plot for column: " + column.getLabel());
                        plotBenfordDistribution(benfordCounts.get(column.getLabel()));
                    }
                }
            }
        }
            return benfordCounts;
        }

    }
