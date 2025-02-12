from collections import defaultdict

def maximumSum(nums: list[int]) -> int:
        sum_to_max = defaultdict(int)  # Stores the max num seen for each digit sum
        max_sum = -1  # Tracks the max pair sum

        for num in nums:
            this_sum = sum(int(digit) for digit in str(num))  # Compute digit sum
            
            if this_sum in sum_to_max:  # If another num with same digit sum exists
                max_sum = max(max_sum, sum_to_max[this_sum] + num)  # Update max pair sum
            
            sum_to_max[this_sum] = max(sum_to_max[this_sum], num)  # Store max num for this digit sum

        return max_sum

#nums = [18,43,36,13,7]
#nums = [10,12,19,14]
nums = [229,398,269,317,420,464,491,218,439,153,482,169,411,93,147,50,347,210,251,366,401]

print (maximumSum(nums))



