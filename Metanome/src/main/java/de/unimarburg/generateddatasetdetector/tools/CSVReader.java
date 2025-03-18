package de.unimarburg.generateddatasetdetector.tools;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.InputStream;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class CSVReader {
    /**
     * Reads a CSV file from the resources folder and returns its content as a list of strings.
     *
     * @param fileName The name of the CSV file located in the resources directory.
     * @return A list of strings representing each line in the CSV file.
     * @throws Exception if the file is not found or cannot be read.
     */
    public static Iterable<CSVRecord> readCSVFromResources(String fileName) throws Exception {
        InputStream inputStream = CSVReader.class.getClassLoader().getResourceAsStream(fileName);
        if (inputStream == null) {
            throw new Exception("File not found: " + fileName);
        }
        Reader reader = new java.io.InputStreamReader(inputStream);
        Iterable<CSVRecord> records = CSVFormat.DEFAULT
                .withHeader()
                .parse(reader);
        return records;
    }

    /**
     * Reads a CSV file from a relative or absolute path and returns its content as a list of strings.
     *
     * @param filePath The path to the CSV file.
     * @return A list of strings representing each line in the CSV file.
     * @throws Exception if the file is not found or cannot be read.
     */
    public static List<String> readCSVFromPath(String filePath) throws Exception {
        return Files.readAllLines(Paths.get(filePath));
    }


}
