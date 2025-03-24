* classID:


A number that matches a sound.
# An example of an animal sound
labels = {
    0: "bee_buzz",
    1: "elephant_trumpet",
    2: "kitten_bark",
    3: "dog_bark",
    4: "woodpecker_peck",
    5: "sloth_snore",
    6: "flamingo_alarm",
    7: "kangaroo_thump",
    8: "wolf_howl",
    9: "nightingale_song"
}

# Obtain animal tags based on predictions
predicted_class = 8
print(animal_labels[predicted_class])  # out put: wolf_howl

# 
filename = "100263-2-0-117.wav"
parts = filename.split("-")
recording_id = parts[0]     # "100263"
class_id = parts[1]         # "2"----"kitten_bark"
subclip_params = parts[2:4] # ["0", "117"] sound begin and end