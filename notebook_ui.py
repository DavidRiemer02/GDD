# notebook_ui.py
from ipywidgets import FileUpload, VBox, Label, Textarea, Button
import os
import zipfile
import io
import json
import shutil

def upload_widget(config):
    java_exe = config["java_exe"]
    test_base_dir_real = config["test_base_dir"]
    metanome_jar = config["metanome_jar"]
    java_memory = config["java_memory"]

    # Ensure required folders exist
    os.makedirs("UserData/metanomeResults", exist_ok=True)
    os.makedirs(test_base_dir_real, exist_ok=True)
    os.makedirs("UserData/archive", exist_ok=True)

    file_uploader = FileUpload(accept=".csv,.zip", multiple=False)
    upload_status = Label(value="")
    intro_label = Label(value="Upload your dataset (CSV or ZIP containing CSV files):")
    def handle_upload(change):
        upload_status.value = ""
        for fileinfo in file_uploader.value:
            filename = fileinfo["name"]
            content = fileinfo["content"]
            save_path = os.path.join(test_base_dir_real, filename)

            if filename.endswith(".zip"):
                try:
                    # Step 1: Temporarily save zip to real data folder
                    with open(save_path, "wb") as f:
                        f.write(content)

                    # Step 2: Extract it
                    with zipfile.ZipFile(save_path, 'r') as zip_ref:
                        zip_ref.extractall(test_base_dir_real)

                    # Step 3: Move the zip file to archive
                    archive_path = os.path.join("UserData/archive", filename)
                    shutil.move(save_path, archive_path)

                    upload_status.value = f"Extracted '{filename}' to '{test_base_dir_real}' and moved ZIP to 'UserData/archive/'"
                except zipfile.BadZipFile:
                    upload_status.value = f"Failed to extract '{filename}': not a valid ZIP file"
            else:
                # Directly save .csv
                with open(save_path, "wb") as f:
                    f.write(content)
                upload_status.value = f"Uploaded '{filename}' to '{test_base_dir_real}'"

    file_uploader.observe(handle_upload, names='value')
    return VBox([intro_label, file_uploader, upload_status])

def config_editor(config_path="config.json"):
    """Creates a widget for editing and saving the JSON config file."""

    # Load existing config (or initialize empty)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_text = f.read()
    else:
        config_text = "{}"

    # Create UI elements
    config_area = Textarea(
        value=config_text,
        placeholder='Edit JSON config here...',
        description='Config:',
        layout={'width': '100%', 'height': '200px'}
    )

    save_button = Button(description="Save Config", button_style="success")
    status_label = Label(value="")

    # Save function
    def save_config(b):
        try:
            new_config = json.loads(config_area.value)
            with open(config_path, "w") as f:
                json.dump(new_config, f, indent=4)
            status_label.value = "Config saved successfully!"
        except json.JSONDecodeError as e:
            status_label.value = f"JSON Error: {str(e)}"

    save_button.on_click(save_config)
    return VBox([config_area, save_button, status_label])
