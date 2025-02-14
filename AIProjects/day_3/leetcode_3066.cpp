class Solution {
public:
    int minOperations(vector<int>& nums, int k) {
        
    priority_queue<long, vector<long>, greater<long>> minHeap(nums.begin(), nums.end());  // Min-heap
    int num_ops = 0;

    while (true) {
        long x = minHeap.top();  // Smallest element
        if (x >= k) return num_ops;  // Stop condition
        minHeap.pop();  // Remove smallest

        //if (minHeap.empty()) return -1;  // Edge case: Not enough elements left

        long y = minHeap.top();  // Second smallest element
        minHeap.pop();  // Remove second smallest

        // Combine and push new element back
        minHeap.push(min(x, y) * 2 + max(x, y));
        num_ops++;
    }


    }
};
