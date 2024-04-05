from src.Utilities.color_space_values import color_space_values


class Video_Results:
    def __init__(self, nitrite: color_space_values = color_space_values(),
                 ketone: color_space_values = color_space_values(),
                 glucose: color_space_values = color_space_values(),
                 blood: color_space_values = color_space_values(),
                 protein: color_space_values = color_space_values(),
                 specific_gravity: color_space_values = color_space_values(),
                 leukocytes: color_space_values = color_space_values(),
                 bilirubin: color_space_values = color_space_values(),
                 urobilinogen: color_space_values = color_space_values(),
                 ph: color_space_values = color_space_values()):
        self.nitrite = nitrite
        self.ketone = ketone
        self.glucose = glucose
        self.blood = blood
        self.protein = protein
        self.specific_gravity = specific_gravity
        self.leukocytes = leukocytes
        self.bilirubin = bilirubin
        self.ph = ph
        self.urobilinogen = urobilinogen

    def update_results_from_dictionary(self, test_results_by_test_name: dict[str, color_space_values]):
        self.nitrite = test_results_by_test_name['TEST-NITRITE'] if 'TEST-NITRITE' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.ketone = test_results_by_test_name['TEST-KETONE'] if 'TEST-KETONE' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.glucose = test_results_by_test_name['TEST-GLUCOSE'] if 'TEST-GLUCOSE' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.blood = test_results_by_test_name['TEST-BLOOD'] if 'TEST-BLOOD' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.protein = test_results_by_test_name['TEST-PROTEIN'] if 'TEST-PROTEIN' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.specific_gravity = test_results_by_test_name['TEST-SPECIFIC_GRAVITY'] if ('TEST-SPECIFIC_GRAVITY' in
                                                                                       test_results_by_test_name) else (
            color_space_values(0, 0, 0, 0))
        self.leukocytes = test_results_by_test_name['TEST-LEUKOCYTES'] if 'TEST-LEUKOCYTES' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.bilirubin = test_results_by_test_name['TEST-BILIRUBIN'] if 'TEST-BILIRUBIN' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.ph = test_results_by_test_name['TEST-PH'] if 'TEST-PH' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.urobilinogen = test_results_by_test_name['TEST-UROBILINOGEN'] if 'TEST-UROBILINOGEN' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
