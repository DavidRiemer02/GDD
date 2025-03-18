package de.unimarburg.generateddatasetdetector;

import de.unimarburg.generateddatasetdetector.algorithms.Benford;
import de.unimarburg.generateddatasetdetector.algorithms.DatasetAuthenticityAnalyzer;
import de.unimarburg.generateddatasetdetector.algorithms.ZipfsLaw;
import de.unimarburg.generateddatasetdetector.data.Database;
import de.unimarburg.generateddatasetdetector.data.Table;
import de.unimarburg.generateddatasetdetector.data.metadata.dependency.FunctionalDependency;
import de.unimarburg.generateddatasetdetector.data.metadata.dependency.InclusionDependency;
import de.unimarburg.generateddatasetdetector.data.metadata.dependency.MetanomeImpl;
import de.unimarburg.generateddatasetdetector.data.metadata.dependency.UniqueColumnCombination;
import org.apache.commons.csv.CSVRecord;
import de.unimarburg.generateddatasetdetector.tools.CSVReader;

import java.util.List;
import java.util.Map;

public class ZipsBenfordMain {
private static final String HORSEPATH = "data/horse";
    private static final String IRISPATH = "data/iris";
    private static final String PROTPATH = "data/uniprot";
    private static final String STUDENTSPERFORMANCEPATH = "data/StudentsPerformance";
    private static final String CVAP = "data/StudentsPerformance";
    private static final String CREDITCARD = "data/creditcard";
    private static final String BENFORDZIPFSDATA = "data/BenfordZipsDatasets";
    private static final String HARRYPOTTER1 = "data/HARRYPOTTER";



    public static void main(String[] args) {
        try {
            // Example 1: Read a file from resources and run Benfords Law
            System.out.println("Reading from resources:");
            Database db = new Database(IRISPATH);
            System.out.println("Data successfully read Database: " + db.getName());
            System.out.println("Done with reading the files \n" +
                    "--------------------------------------");
            System.out.println("Calculating Benfords Law:");
            //Map<String, Map<Integer, Integer>> BENFORDresults = Benford.calcBenfordFromTable(db.getTables());
            System.out.println("Done with Benfords Law \n" +
                    "--------------------------------------");
            System.out.println("Calculating Zipfs Law:");
            //Map<String, Double> ZIPFresults = ZipfsLaw.calcZipfWithChiSquareFromTable(db.getTables());
            System.out.println("Done with Zipfs Law \n" +
                    "--------------------------------------");
            System.out.println("Evaluating Dataset Authenticity:");
            //DatasetAuthenticityAnalyzer.evaluateDatasetAuthenticity(BENFORDresults, ZIPFresults);
            System.out.println("Done with evaluating Dataset Authenticity \n" +
                    "--------------------------------------");

            List<FunctionalDependency> FDresults = MetanomeImpl.executeFD(db.getTables());
            for (FunctionalDependency fd : FDresults) {
                System.out.println(fd.toString());
            }

            List<UniqueColumnCombination> UCCresults = MetanomeImpl.executeUCC(db.getTables());
            System.out.println("Number of UCCs: " + UCCresults.size());

            List<InclusionDependency> INDresults = MetanomeImpl.executeIND(db.getTables());
            System.out.println("Number of INDs: " + INDresults.size());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}