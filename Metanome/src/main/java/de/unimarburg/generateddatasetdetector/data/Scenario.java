package de.unimarburg.generateddatasetdetector.data;

import de.unimarburg.generateddatasetdetector.data.metadata.ScenarioMetadata;
import de.unimarburg.generateddatasetdetector.utils.Configuration;
import de.unimarburg.generateddatasetdetector.utils.InputReader;
import de.unimarburg.generateddatasetdetector.utils.StringUtils;
import lombok.Data;

import java.io.File;

@Data
public class Scenario {
    private final String path;
    private final String name;
    private Database sourceDatabase;
    private Database targetDatabase;
    private ScenarioMetadata metadata;

    public Scenario(String path) {
        this.path = path;
        this.name = StringUtils.getFolderName(path);
        this.sourceDatabase = new Database(this.path + File.separator + Configuration.getInstance().getDefaultSourceDatabaseDir());
        this.targetDatabase = new Database(this.path + File.separator + Configuration.getInstance().getDefaultTargetDatabaseDir());
        // TODO: read dependencies on demand
        if (Configuration.getInstance().isReadDependencies()) {
            this.metadata = InputReader.readScenarioMetadata(this.path, sourceDatabase, targetDatabase);
        }
    }
}