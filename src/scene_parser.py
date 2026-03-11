import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

input_file = "moviesum_train.jsonl"
output_file = "moviesum_scene_chunks.json"

scene_chunks = []

with open(input_file, "r", encoding="utf-8") as f:

    for line in tqdm(f, desc="Processing movies"):

        movie = json.loads(line)

        movie_name = movie.get("movie_name")
        imdb_id = movie.get("imdb_id")
        script_xml = movie.get("script")

        if not script_xml:
            continue

        try:
            root = ET.fromstring(script_xml)
        except:
            continue

        scene_counter = 1
        current_character = ""

        for scene in root.findall("scene"):

            stage_direction = ""
            scene_description = ""
            dialogues = []

            for elem in scene:

                tag = elem.tag
                text = elem.text.strip() if elem.text else ""

                if tag == "stage_direction":
                    stage_direction = text

                elif tag == "scene_description":
                    scene_description = text

                elif tag == "character":
                    current_character = text

                elif tag == "dialogue":
                    if current_character:
                        dialogues.append(f"{current_character}: {text}")
                    else:
                        dialogues.append(text)

            # Create scene text
            parts = []

            if stage_direction:
                parts.append(stage_direction)

            if scene_description:
                parts.append(scene_description)

            if dialogues:
                parts.append("\n".join(dialogues))

            scene_text = "\n".join(parts)

            if len(scene_text) < 30:
                continue

            scene_data = {
                "id": f"{imdb_id}_scene_{scene_counter}",
                "text": scene_text,
                "metadata": {
                    "movie_name": movie_name,
                    "imdb_id": imdb_id,
                    "scene_id": scene_counter
                }
            }

            scene_chunks.append(scene_data)

            scene_counter += 1


# Save all scenes
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(scene_chunks, f, indent=2, ensure_ascii=False)


print("Scene extraction completed.")
print("Total scenes:", len(scene_chunks))
print("Saved to:", output_file)