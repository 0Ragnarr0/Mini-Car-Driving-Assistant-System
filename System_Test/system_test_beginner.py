#config safety level requirements for the system
#Refer to system_test_notes.txt for more details

class SafetyConfig():
    safe_distance_normal_threshold = 25 #in meters

    def __init__(self, distance_to_obstacle):
        self.distance_to_obstacle = distance_to_obstacle
    def check_safety(self):
        if self.distance_to_obstacle >= self.safe_distance_normal_threshold:
            return f"Pass - Safe distance maintained, {self.distance_to_obstacle} meters apart."
        else:
            error_distance = self.safe_distance_normal_threshold - self.distance_to_obstacle
            return f"Fail - Unsafe distance to obstacle.{error_distance} meters below normal threshold"

#Example usage
if __name__ == "__main__":
    #Test case 1: Safe distance maintained
    test1 = SafetyConfig(30)
    print(test1.check_safety())  # Expected: Pass - Safe distance maintained

    #Test case 2: Unsafe distance to obstacle
    test2 = SafetyConfig(20)
    print(test2.check_safety())  # Expected: Fail - Unsafe distance to obstacle, -5 meters below normal threshold