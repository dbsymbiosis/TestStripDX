from src.Utilities.RGBValue import RGBValue


class Video_Results:
    def __init__(self,nitrite: RGBValue = RGBValue(0, 0, 0, 0),
                 ketone: RGBValue = RGBValue(0, 0, 0, 0),
                 glucose: RGBValue = RGBValue(0, 0, 0, 0),
                 blood: RGBValue = RGBValue(0, 0, 0, 0),
                 protein: RGBValue = RGBValue(0, 0, 0, 0),
                 specific_gravity: RGBValue = RGBValue(0, 0, 0, 0),
                 leukocytes: RGBValue = RGBValue(0, 0, 0, 0),
                 bilirubin: RGBValue = RGBValue(0, 0, 0, 0),
                 urobilinogen: RGBValue = RGBValue(0, 0, 0, 0),
                 ph: RGBValue = RGBValue(0, 0, 0, 0)):
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

    def update_results_from_dictionary(self,test_results_by_test_name:dict[str,RGBValue]):
        self.nitrite = test_results_by_test_name['TEST-NITRITE'] if test_results_by_test_name['TEST-NITRITE'] else (
                            RGBValue(0, 0, 0, 0))
        self.ketone = test_results_by_test_name['TEST-NITRITE'] if test_results_by_test_name['TEST-NITRITE'] else (
                            RGBValue(0, 0, 0, 0))
        self.glucose = test_results_by_test_name['TEST-NITRITE'] if test_results_by_test_name['TEST-NITRITE'] else (
                            RGBValue(0, 0, 0, 0))
        self.blood = test_results_by_test_name['TEST-NITRITE'] if test_results_by_test_name['TEST-NITRITE'] else (
                            RGBValue(0, 0, 0, 0))
        self.protein = test_results_by_test_name['TEST-NITRITE'] if test_results_by_test_name['TEST-NITRITE'] else (
                            RGBValue(0, 0, 0, 0))
        self.specific_gravity = test_results_by_test_name['TEST-NITRITE'] if test_results_by_test_name['TEST-NITRITE'] \
                                    else RGBValue(0, 0, 0, 0)
        self.leukocytes = test_results_by_test_name['TEST-NITRITE'] if test_results_by_test_name['TEST-NITRITE'] else (
                            RGBValue(0, 0, 0, 0))
        self.bilirubin = test_results_by_test_name['TEST-NITRITE'] if test_results_by_test_name['TEST-NITRITE'] else (
                            RGBValue(0, 0, 0, 0))
        self.ph = test_results_by_test_name['TEST-NITRITE'] if test_results_by_test_name['TEST-NITRITE'] else (
                            RGBValue(0, 0, 0, 0))
        self.urobilinogen = test_results_by_test_name['TEST-UROBILINOGEN'] if test_results_by_test_name['TEST-UROBILINOGEN'] else (
                            RGBValue(0, 0, 0, 0))

