package de.unimarburg.generateddatasetdetector.similarity;

public interface SimilarityMeasure<DataType> {
    float compare(DataType source, DataType target);
}
