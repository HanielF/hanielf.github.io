---
title: LeetCode-019-Remove Nth Node From End of List
tags:
  - LeetCode
  - Algorithm
  - LinkedList
  - TwoPointer
  - Medium
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-07-21 23:51:02
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/submissions/)   
Given a linked list, remove the n-th node from the end of list and return its head.

Note:
Given n will always be valid.

### Examples:
**Input:**1->2->3->4->5, and n = 2.
After removing the second node from the end, the linked list becomes 1->2->3->5.
{% endnote %}
<!--more-->

## Solutions
-
第一种方法是两次遍历,第一次计算出一共多少个节点,然后算出应该向后移动多少次,
找到那个节点, 删掉
- 第二种方法是使用双指针一次遍历, 两个指针初始化为head,
后一个指针先跑n步, 然后当后一个指针到达最后一个位置时,
前一个指针是倒数第n+1个, 删掉第n个位置就可以

## C++ Codes
双指针
```C++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* p = head;
        ListNode* pre=head;
        for(n;n>=1;n--) p=p->next;
        if(p==NULL)return head->next;
        while(p->next!=NULL){
            p=p->next;
            pre=pre->next;
        }
        pre->next=pre->next->next;
        return head;
    }
};
```

## Python Codes

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if not head:return 
        dummy = ListNode(0)
        dummy.next = head
        fast = dummy
        while n:
            fast = fast.next
            n -= 1
        slow = dummy
        while fast and fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next
```

## 总结
- 双指针另一个用法: 链表固定相差n个位置移动, 查找节点

------
