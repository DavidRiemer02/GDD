package de.unimarburg.generateddatasetdetector.views.upload;


import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.router.Menu;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;
import de.unimarburg.generateddatasetdetector.tools.ExcelUploader;
import org.vaadin.lineawesome.LineAwesomeIconUrl;

import java.io.File;

@PageTitle("Upload File")
@Route("uploadFile")
@Menu(order = 1, icon = LineAwesomeIconUrl.FILE)
public class UploadFile extends HorizontalLayout {

    public UploadFile() {
        add(getContent());
    }

    private VerticalLayout getContent() {
        VerticalLayout body = new VerticalLayout();
        File uploadFolder = getUploadFolder();
        ExcelUploader uploadArea = new ExcelUploader(uploadFolder);
        uploadArea.getUploadField().addSucceededListener(e -> {
            uploadArea.hideErrorField();
        });
        body.add(uploadArea);
        return body;
    }

    private static File getUploadFolder() {
        File folder = new File("uploaded-files");
        if (!folder.exists()) {
            folder.mkdirs();
        }
        return folder;
    }
}
