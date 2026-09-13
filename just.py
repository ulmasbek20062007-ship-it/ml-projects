class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxDepth(root):
    if

# build a small tree:
#       1
#      / \
#     2   3
#    /
#   4
root = TreeNode(1, TreeNode(2, TreeNode(4)), TreeNode(3))

print(maxDepth(root))  # expected: 3