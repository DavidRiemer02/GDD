package de.unimarburg.generateddatasetdetector.similarity.string;

import de.unimarburg.generateddatasetdetector.similarity.SimilarityMeasure;
import org.apache.commons.text.similarity.LevenshteinDistance;

public class Levenshtein implements SimilarityMeasure<String> {

    public Levenshtein() {
    }

    @Override
    public float compare(String s1, String s2) {
        int dist = LevenshteinDistance.getDefaultInstance().apply(s1, s2);
        return 1 - (float)dist / Math.max(s1.length(), s2.length());
    }
}
