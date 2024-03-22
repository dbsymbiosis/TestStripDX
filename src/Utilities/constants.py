from enum import Enum

from src.Utilities.color_space_values import color_space_values

YOLOV8_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
standards_color_space_values = {
    'Red': color_space_values(255, 0, 0, 85, 54.29, 80.81, 69.89, 0, 100
                              , 100, 0),
    'Green': color_space_values(0, 255, 0, 85, 87.82, -79.29, 80.99, 62, 0,
                                100, 0),
    'Blue': color_space_values(0, 0, 255, 85, 29.57, 28.30, -112.03,
                               88, 77, 0, 0)
}

test_timings = {
    'TEST-GLUCOSE': 30,
    'TEST-BILIRUBIN': 30,
    'TEST-KETONE': 40,
    'TEST-SPECIFIC_GRAVITY': 45,
    'TEST-BLOOD': 60,
    'TEST-PH': 60,
    'TEST-PROTEIN': 60,
    'TEST-UROBILINOGEN': 60,
    'TEST-NITRITE': 60,
    'TEST-LEUKOCYTES': 119
}
hue_shifts = []
shift = 0
while shift < 360:
    hue_shifts.append(shift)
    shift += 30
csv_headers = ['Video-Name']
for shift in hue_shifts:
    headers_for_shift = ['TEST-BILIRUBIN-Red'+'.shift'+str(shift), 'TEST-BILIRUBIN-Green'+'.shift'+str(shift), 'TEST-BILIRUBIN-Blue'+'.shift'+str(shift),
                         'TEST-BILIRUBIN-RGB-MeanScore'+'.shift'+str(shift),'TEST-BILIRUBIN-L-star'+'.shift'+str(shift),'TEST-BILIRUBIN-a-star'+'.shift'+str(shift),
                         'TEST-BILIRUBIN-b-star' + '.shift' + str(shift),'TEST-BILIRUBIN-cyan'+'.shift'+str(shift),'TEST-BILIRUBIN-yellow'+'.shift'+str(shift),
                         'TEST-BILIRUBIN-magenta' + '.shift' + str(shift),'TEST-BILIRUBIN-key-black'+'.shift'+str(shift),
                         'TEST-BLOOD-Red'+'.shift'+str(shift), 'TEST-BLOOD-Green'+'.shift'+str(shift), 'TEST-BLOOD-Blue'+'.shift'+str(shift),
                         'TEST-BLOOD-RGB-MeanScore'+'.shift'+str(shift),'TEST-BLOOD-L-star'+'.shift'+str(shift),'TEST-BLOOD-a-star'+'.shift'+str(shift),
                         'TEST-BLOOD-b-star' + '.shift' + str(shift),'TEST-BLOOD-cyan'+'.shift'+str(shift),'TEST-BLOOD-yellow'+'.shift'+str(shift),
                         'TEST-BLOOD-magenta' + '.shift' + str(shift),'TEST-BLOOD-key-black'+'.shift'+str(shift),
                         'TEST-GLUCOSE-Red'+'.shift'+str(shift), 'TEST-GLUCOSE-Green'+'.shift'+str(shift), 'TEST-GLUCOSE-Blue'+'.shift'+str(shift),
                         'TEST-GLUCOSE-RGB-MeanScore'+'.shift'+str(shift),'TEST-GLUCOSE-L-star'+'.shift'+str(shift),'TEST-GLUCOSE-a-star'+'.shift'+str(shift),
                         'TEST-GLUCOSE-b-star' + '.shift' + str(shift),'TEST-GLUCOSE-cyan'+'.shift'+str(shift),'TEST-GLUCOSE-yellow'+'.shift'+str(shift),
                         'TEST-GLUCOSE-magenta' + '.shift' + str(shift),'TEST-GLUCOSE-key-black'+'.shift'+str(shift),
                         'TEST-KETONE-Red'+'.shift'+str(shift), 'TEST-KETONE-Green'+'.shift'+str(shift), 'TEST-KETONE-Blue'+'.shift'+str(shift),
                         'TEST-KETONE-RGB-MeanScore'+'.shift'+str(shift),'TEST-KETONE-L-star'+'.shift'+str(shift),'TEST-KETONE-a-star'+'.shift'+str(shift),
                         'TEST-KETONE-b-star' + '.shift' + str(shift),'TEST-KETONE-cyan'+'.shift'+str(shift),'TEST-KETONE-yellow'+'.shift'+str(shift),
                         'TEST-KETONE-magenta' + '.shift' + str(shift),'TEST-KETONE-key-black'+'.shift'+str(shift),
                         'TEST-LEUKOCYTES-Red'+'.shift'+str(shift), 'TEST-LEUKOCYTES-Green'+'.shift'+str(shift), 'TEST-LEUKOCYTES-Blue'+'.shift'+str(shift),
                         'TEST-LEUKOCYTES-RGB-MeanScore'+'.shift'+str(shift),'TEST-LEUKOCYTES-L-star'+'.shift'+str(shift),'TEST-LEUKOCYTES-a-star'+'.shift'+str(shift),
                         'TEST-LEUKOCYTES-b-star' + '.shift' + str(shift),'TEST-LEUKOCYTES-cyan'+'.shift'+str(shift),'TEST-LEUKOCYTES-yellow'+'.shift'+str(shift),
                         'TEST-LEUKOCYTES-magenta' + '.shift' + str(shift),'TEST-LEUKOCYTES-key-black'+'.shift'+str(shift),
                         'TEST-NITRITE-Red'+'.shift'+str(shift), 'TEST-NITRITE-Green'+'.shift'+str(shift), 'TEST-NITRITE-Blue'+'.shift'+str(shift),
                         'TEST-NITRITE-RGB-MeanScore'+'.shift'+str(shift),'TEST-NITRITE-L-star'+'.shift'+str(shift),'TEST-NITRITE-a-star'+'.shift'+str(shift),
                         'TEST-NITRITE-b-star' + '.shift' + str(shift),'TEST-NITRITE-cyan'+'.shift'+str(shift),'TEST-NITRITE-yellow'+'.shift'+str(shift),
                         'TEST-NITRITE-magenta' + '.shift' + str(shift),'TEST-NITRITE-key-black'+'.shift'+str(shift),
                         'TEST-PH-Red'+'.shift'+str(shift), 'TEST-PH-Green'+'.shift'+str(shift), 'TEST-PH-Blue'+'.shift'+str(shift),
                         'TEST-PH-RGB-MeanScore'+'.shift'+str(shift),'TEST-PH-L-star'+'.shift'+str(shift),'TEST-PH-a-star'+'.shift'+str(shift),
                         'TEST-PH-b-star' + '.shift' + str(shift),'TEST-PH-cyan'+'.shift'+str(shift),'TEST-PH-yellow'+'.shift'+str(shift),
                         'TEST-PH-magenta' + '.shift' + str(shift),'TEST-PH-key-black'+'.shift'+str(shift),
                         'TEST-PROTEIN-Red'+'.shift'+str(shift), 'TEST-PROTEIN-Green'+'.shift'+str(shift), 'TEST-PROTEIN-Blue'+'.shift'+str(shift),
                         'TEST-PROTEIN-RGB-MeanScore'+'.shift'+str(shift),'TEST-PROTEIN-L-star'+'.shift'+str(shift),'TEST-PROTEIN-a-star'+'.shift'+str(shift),
                         'TEST-PROTEIN-b-star' + '.shift' + str(shift),'TEST-PROTEIN-cyan'+'.shift'+str(shift),'TEST-PROTEIN-yellow'+'.shift'+str(shift),
                         'TEST-PROTEIN-magenta' + '.shift' + str(shift),'TEST-PROTEIN-key-black'+'.shift'+str(shift),
                         'TEST-SPECIFIC_GRAVITY-Red'+'.shift'+str(shift), 'TEST-SPECIFIC_GRAVITY-Green'+'.shift'+str(shift), 'TEST-SPECIFIC_GRAVITY-Blue'+'.shift'+str(shift),
                         'TEST-SPECIFIC_GRAVITY-RGB-MeanScore'+'.shift'+str(shift),'TEST-SPECIFIC_GRAVITY-L-star'+'.shift'+str(shift),'TEST-SPECIFIC_GRAVITY-a-star'+'.shift'+str(shift),
                         'TEST-SPECIFIC_GRAVITY-b-star' + '.shift' + str(shift),'TEST-SPECIFIC_GRAVITY-cyan'+'.shift'+str(shift),'TEST-SPECIFIC_GRAVITY-yellow'+'.shift'+str(shift),
                         'TEST-SPECIFIC_GRAVITY-magenta' + '.shift' + str(shift),'TEST-SPECIFIC_GRAVITY-key-black'+'.shift'+str(shift),
                         'TEST-UROBILINOGEN-Red'+'.shift'+str(shift), 'TEST-UROBILINOGEN-Green'+'.shift'+str(shift), 'TEST-UROBILINOGEN-Blue'+'.shift'+str(shift),
                         'TEST-UROBILINOGEN-RGB-MeanScore'+'.shift'+str(shift),'TEST-UROBILINOGEN-L-star'+'.shift'+str(shift),'TEST-UROBILINOGEN-a-star'+'.shift'+str(shift),
                         'TEST-UROBILINOGEN-b-star' + '.shift' + str(shift),'TEST-UROBILINOGEN-cyan'+'.shift'+str(shift),'TEST-UROBILINOGEN-yellow'+'.shift'+str(shift),
                         'TEST-UROBILINOGEN-magenta' + '.shift' + str(shift),'TEST-UROBILINOGEN-key-black'+'.shift'+str(shift)]
    csv_headers.extend(headers_for_shift)
