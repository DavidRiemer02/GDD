package de.unimarburg.generateddatasetdetector.similarity.set;

import de.unimarburg.generateddatasetdetector.similarity.SimilarityMeasure;
import de.unimarburg.generateddatasetdetector.utils.SetUtils;

import java.util.Set;

public class Dice<T> implements SimilarityMeasure<Set<T>> {
    @Override
    public float compare(Set<T> source, Set<T> target) {
        int sizeSource = source.size();
        int sizeTarget = target.size();
        Set<T> intersection = SetUtils.intersection(source, target);
        int sizeIntersection = intersection.size();
        return (float) 2 * sizeIntersection / (sizeSource + sizeTarget);
    }
}
