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
        self.nitrite = test_results_by_test_name['Test-Nitrite'] if 'Test-Nitrite' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.ketone = test_results_by_test_name['Test-Ketone'] if 'Test-Ketone' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.glucose = test_results_by_test_name['Test-Glucose'] if 'Test-Glucose' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.blood = test_results_by_test_name['Test-Blood'] if 'Test-Blood' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.protein = test_results_by_test_name['Test-Protein'] if 'Test-Protein' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.specific_gravity = test_results_by_test_name['Test-Specific_Gravity'] if ('Test-Specific_Gravity' in
                                                                                       test_results_by_test_name) else (
            color_space_values(0, 0, 0, 0))
        self.leukocytes = test_results_by_test_name['Test-Leukocytes'] if 'Test-Leukocytes' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.bilirubin = test_results_by_test_name['Test-Bilirubin'] if 'Test-Bilirubin' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.ph = test_results_by_test_name['Test-PH'] if 'Test-PH' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
        self.urobilinogen = test_results_by_test_name['Test-Urobilinogen'] if 'Test-Urobilinogen' in test_results_by_test_name else (
            color_space_values(0, 0, 0, 0))
