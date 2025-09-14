import modal
import os
import shutil

# Your Modal volume name
VOLUME_NAME = "embedding-model-vol"

# List only the exact files/directories you want deleted
FILES_TO_DELETE = [
    "faiss_bge_index_new",
    "faiss_bge_index_newdocs",
    "faiss_index", 
    "hackbattle_doc_cleaned.txt",
    "hackbattle_docs",
    "hackbattle_docs_new",
    "HackBattle_Info.txt",
    "models"
]
 
app = modal.App("cleanup-script")

# Lookup the volume object
volume = modal.Volume.lookup(VOLUME_NAME, create_if_missing=False)

# Mount the volume at /vol inside the container
@app.function(volumes={volume: "/vol"})
def cleanup():
    base_path = "/vol"
    
    for path in FILES_TO_DELETE:
        full_path = os.path.join(base_path, path)
        if os.path.exists(full_path):
            if os.path.isfile(full_path):
                os.remove(full_path)
                print(f"Deleted file: {path}")
            else:
                shutil.rmtree(full_path)
                print(f"Deleted directory: {path}")
        else:
            print(f"Not found (skipped): {path}")

    print("Cleanup complete ✅")

if __name__ == "__main__":
    with app.run():
        cleanup()