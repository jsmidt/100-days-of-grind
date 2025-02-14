class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        heapq.heapify(nums)  # Convert list into a min-heap (O(N))
        num_ops = 0

        while nums[0] < k:  # Keep processing until smallest element >= k
            if len(nums) < 2:  # Edge case: if fewer than 2 elements remain
                return -1

            # Extract the two smallest elements (O(log N) each)
            x = heapq.heappop(nums)
            y = heapq.heappop(nums)

            # Combine them with the given formula and push back to heap (O(log N))
            heapq.heappush(nums, min(x, y) * 2 + max(x, y))
            num_ops += 1

        return num_ops

