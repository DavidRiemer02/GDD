package de.unimarburg.generateddatasetdetector.algorithms;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class TextPreprocessor {
    private static final Set<String> DEFAULT_STOPWORDS = Set.of(
            "a", "an", "the", "and", "or", "but", "if", "to", "of", "in", "on", "with", "as", "for", "is", "are"
    );

    private final Set<String> stopwords;

    public TextPreprocessor() {
        this.stopwords = new HashSet<>(DEFAULT_STOPWORDS);
    }

    public TextPreprocessor(Set<String> customStopwords) {
        this.stopwords = new HashSet<>(customStopwords);
    }

    /**
     * Processes the text by tokenizing, removing stopwords, and reconstructing the string.
     * @param text The original text.
     * @return The processed text with stopwords removed.
     */
    public String process(String text) {
        if (text == null || text.isEmpty()) {
            return text;
        }

        // Tokenize the text
        List<String> tokens = tokenize(text);

        // Remove stopwords
        List<String> filteredTokens = removeStopwords(tokens);

        // Reconstruct the edited string
        return reconstructString(filteredTokens);
    }

    /**
     * Tokenizes the text into words.
     * @param text The text to tokenize.
     * @return A list of tokens.
     */
    private List<String> tokenize(String text) {
        return Arrays.asList(text.split("\\s+"));
    }

    /**
     * Removes stopwords from a list of tokens.
     * @param tokens The list of tokens.
     * @return A list of tokens with stopwords removed.
     */
    private List<String> removeStopwords(List<String> tokens) {
        return tokens.stream()
                .filter(token -> !stopwords.contains(token.toLowerCase()))
                .collect(Collectors.toList());
    }

    /**
     * Reconstructs a string from a list of tokens.
     * @param tokens The list of tokens.
     * @return The reconstructed string.
     */
    private String reconstructString(List<String> tokens) {
        return String.join(" ", tokens);
    }

}
