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
    'Test-Glucose': 30,
    'Test-Bilirubin': 30,
    'Test-Ketone': 40,
    'Test-Specific_Gravity': 45,
    'Test-Blood': 60,
    'Test-PH': 60,
    'Test-Protein': 60,
    'Test-Urobilinogen': 60,
    'Test-Nitrite': 60,
    'Test-Leukocytes': 119
}
hue_shifts = []
shift = 0
while shift < 360:
    hue_shifts.append(shift)
    shift += 30
csv_headers = ['Video-Name']
for shift in hue_shifts:
    headers_for_shift = ['Test-Bilirubin-Red'+'.shift'+str(shift), 'Test-Bilirubin-Green'+'.shift'+str(shift), 'Test-Bilirubin-Blue'+'.shift'+str(shift),
                         'Test-Bilirubin-RGB-MeanScore'+'.shift'+str(shift),'Test-Bilirubin-L-star'+'.shift'+str(shift),'Test-Bilirubin-a-star'+'.shift'+str(shift),
                         'Test-Bilirubin-b-star' + '.shift' + str(shift),'Test-Bilirubin-cyan'+'.shift'+str(shift),'Test-Bilirubin-yellow'+'.shift'+str(shift),
                         'Test-Bilirubin-magenta' + '.shift' + str(shift),'Test-Bilirubin-key-black'+'.shift'+str(shift),
                         'Test-Blood-Red'+'.shift'+str(shift), 'Test-Blood-Green'+'.shift'+str(shift), 'Test-Blood-Blue'+'.shift'+str(shift),
                         'Test-Blood-RGB-MeanScore'+'.shift'+str(shift),'Test-Blood-L-star'+'.shift'+str(shift),'Test-Blood-a-star'+'.shift'+str(shift),
                         'Test-Blood-b-star' + '.shift' + str(shift),'Test-Blood-cyan'+'.shift'+str(shift),'Test-Blood-yellow'+'.shift'+str(shift),
                         'Test-Blood-magenta' + '.shift' + str(shift),'Test-Blood-key-black'+'.shift'+str(shift),
                         'Test-Glucose-Red'+'.shift'+str(shift), 'Test-Glucose-Green'+'.shift'+str(shift), 'Test-Glucose-Blue'+'.shift'+str(shift),
                         'Test-Glucose-RGB-MeanScore'+'.shift'+str(shift),'Test-Glucose-L-star'+'.shift'+str(shift),'Test-Glucose-a-star'+'.shift'+str(shift),
                         'Test-Glucose-b-star' + '.shift' + str(shift),'Test-Glucose-cyan'+'.shift'+str(shift),'Test-Glucose-yellow'+'.shift'+str(shift),
                         'Test-Glucose-magenta' + '.shift' + str(shift),'Test-Glucose-key-black'+'.shift'+str(shift),
                         'Test-Ketone-Red'+'.shift'+str(shift), 'Test-Ketone-Green'+'.shift'+str(shift), 'Test-Ketone-Blue'+'.shift'+str(shift),
                         'Test-Ketone-RGB-MeanScore'+'.shift'+str(shift),'Test-Ketone-L-star'+'.shift'+str(shift),'Test-Ketone-a-star'+'.shift'+str(shift),
                         'Test-Ketone-b-star' + '.shift' + str(shift),'Test-Ketone-cyan'+'.shift'+str(shift),'Test-Ketone-yellow'+'.shift'+str(shift),
                         'Test-Ketone-magenta' + '.shift' + str(shift),'Test-Ketone-key-black'+'.shift'+str(shift),
                         'Test-Leukocytes-Red'+'.shift'+str(shift), 'Test-Leukocytes-Green'+'.shift'+str(shift), 'Test-Leukocytes-Blue'+'.shift'+str(shift),
                         'Test-Leukocytes-RGB-MeanScore'+'.shift'+str(shift),'Test-Leukocytes-L-star'+'.shift'+str(shift),'Test-Leukocytes-a-star'+'.shift'+str(shift),
                         'Test-Leukocytes-b-star' + '.shift' + str(shift),'Test-Leukocytes-cyan'+'.shift'+str(shift),'Test-Leukocytes-yellow'+'.shift'+str(shift),
                         'Test-Leukocytes-magenta' + '.shift' + str(shift),'Test-Leukocytes-key-black'+'.shift'+str(shift),
                         'Test-Nitrite-Red'+'.shift'+str(shift), 'Test-Nitrite-Green'+'.shift'+str(shift), 'Test-Nitrite-Blue'+'.shift'+str(shift),
                         'Test-Nitrite-RGB-MeanScore'+'.shift'+str(shift),'Test-Nitrite-L-star'+'.shift'+str(shift),'Test-Nitrite-a-star'+'.shift'+str(shift),
                         'Test-Nitrite-b-star' + '.shift' + str(shift),'Test-Nitrite-cyan'+'.shift'+str(shift),'Test-Nitrite-yellow'+'.shift'+str(shift),
                         'Test-Nitrite-magenta' + '.shift' + str(shift),'Test-Nitrite-key-black'+'.shift'+str(shift),
                         'Test-PH-Red'+'.shift'+str(shift), 'Test-PH-Green'+'.shift'+str(shift), 'Test-PH-Blue'+'.shift'+str(shift),
                         'Test-PH-RGB-MeanScore'+'.shift'+str(shift),'Test-PH-L-star'+'.shift'+str(shift),'Test-PH-a-star'+'.shift'+str(shift),
                         'Test-PH-b-star' + '.shift' + str(shift),'Test-PH-cyan'+'.shift'+str(shift),'Test-PH-yellow'+'.shift'+str(shift),
                         'Test-PH-magenta' + '.shift' + str(shift),'Test-PH-key-black'+'.shift'+str(shift),
                         'Test-Protein-Red'+'.shift'+str(shift), 'Test-Protein-Green'+'.shift'+str(shift), 'Test-Protein-Blue'+'.shift'+str(shift),
                         'Test-Protein-RGB-MeanScore'+'.shift'+str(shift),'Test-Protein-L-star'+'.shift'+str(shift),'Test-Protein-a-star'+'.shift'+str(shift),
                         'Test-Protein-b-star' + '.shift' + str(shift),'Test-Protein-cyan'+'.shift'+str(shift),'Test-Protein-yellow'+'.shift'+str(shift),
                         'Test-Protein-magenta' + '.shift' + str(shift),'Test-Protein-key-black'+'.shift'+str(shift),
                         'Test-Specific_Gravity-Red'+'.shift'+str(shift), 'Test-Specific_Gravity-Green'+'.shift'+str(shift), 'Test-Specific_Gravity-Blue'+'.shift'+str(shift),
                         'Test-Specific_Gravity-RGB-MeanScore'+'.shift'+str(shift),'Test-Specific_Gravity-L-star'+'.shift'+str(shift),'Test-Specific_Gravity-a-star'+'.shift'+str(shift),
                         'Test-Specific_Gravity-b-star' + '.shift' + str(shift),'Test-Specific_Gravity-cyan'+'.shift'+str(shift),'Test-Specific_Gravity-yellow'+'.shift'+str(shift),
                         'Test-Specific_Gravity-magenta' + '.shift' + str(shift),'Test-Specific_Gravity-key-black'+'.shift'+str(shift),
                         'Test-Urobilinogen-Red'+'.shift'+str(shift), 'Test-Urobilinogen-Green'+'.shift'+str(shift), 'Test-Urobilinogen-Blue'+'.shift'+str(shift),
                         'Test-Urobilinogen-RGB-MeanScore'+'.shift'+str(shift),'Test-Urobilinogen-L-star'+'.shift'+str(shift),'Test-Urobilinogen-a-star'+'.shift'+str(shift),
                         'Test-Urobilinogen-b-star' + '.shift' + str(shift),'Test-Urobilinogen-cyan'+'.shift'+str(shift),'Test-Urobilinogen-yellow'+'.shift'+str(shift),
                         'Test-Urobilinogen-magenta' + '.shift' + str(shift),'Test-Urobilinogen-key-black'+'.shift'+str(shift)]
    csv_headers.extend(headers_for_shift)
