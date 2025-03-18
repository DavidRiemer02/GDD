package de.unimarburg.generateddatasetdetector.algorithms;

import java.util.Map;

public class DatasetAuthenticityAnalyzer {

    // Method to analyze Benford's Law compliance
    public static double analyzeBenfordsLaw(Map<Integer, Integer> benfordCounts) {
        double[] expectedDistribution = {0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046};
        int total = benfordCounts.values().stream().mapToInt(Integer::intValue).sum();
        double chiSquare = 0.0;

        for (int i = 1; i <= 9; i++) {
            double observed = benfordCounts.getOrDefault(i, 0);
            double expected = total * expectedDistribution[i - 1];
            chiSquare += Math.pow(observed - expected, 2) / expected;
        }

        return chiSquare; // Lower value indicates better compliance
    }



    // Method to analyze Zipf's Law compliance
    public static double analyzeZipfsLaw(Map<String, Integer> wordFrequencies) {
        int rank = 1;
        int total = wordFrequencies.values().stream().mapToInt(Integer::intValue).sum();
        double harmonicSum = harmonicSum(wordFrequencies.size());
        double chiSquare = 0.0;

        for (int frequency : wordFrequencies.values()) {
            double expected = (total / harmonicSum) / rank++;
            chiSquare += Math.pow(frequency - expected, 2) / expected;
        }

        return chiSquare; // Lower value indicates better compliance
    }

    // Helper method to calculate the harmonic sum
    private static double harmonicSum(int n) {
        double sum = 0.0;
        for (int i = 1; i <= n; i++) {
            sum += 1.0 / i;
        }
        return sum;
    }

    // Method to evaluate overall dataset authenticity
    public static void evaluateDatasetAuthenticity(
            Map<String, Map<Integer, Integer>> benfordResults,
            Map<String, Double> zipfResults) {
        for (Map.Entry<String, Map<Integer, Integer>> entry : benfordResults.entrySet()) {
            String columnName = entry.getKey();
            double chiSquare = analyzeBenfordsLaw(entry.getValue());
            System.out.printf("Benford's Law for column '%s': Chi-Square = %.2f\n", columnName, chiSquare);
        }

        // Zipf's Law Analysis
        for (Map.Entry<String, Double> entry : zipfResults.entrySet()) {
            String columnName = entry.getKey();
            double chiSquare = entry.getValue();
            System.out.printf("Zipf's Law for column '%s': Chi-Square = %.2f\n", columnName, chiSquare);
        }
        System.out.println("-------------------------------------------------");
        System.out.println("Lower Chi-Square values indicate better compliance with expected laws.");
    }
}
