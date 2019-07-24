---
title: LeetCode-024-Swap Nodes in Pairs
tags:
  - LeetCode
  - Swap
  - LinkedList
  - Medium
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-07-24 01:15:46
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/swap-nodes-in-pairs/submissions/)   
Given a linked list, swap every two adjacent nodes and return its head.

You may **not** modify the values in the list's nodes, only nodes itself may be changed.

### Examples:
**Input:** 1-2-3-4
**Output:** 2-1-4-3
{% endnote %}
<!--more-->

## Solutions
- 很直接的, 每两个节点交换一下顺序, 可以递归但是并不需要.
- 指针之间的交换要注意不要漏了某个next的赋值, 可能会导致出现环
- 多指针的题目宁愿多弄几个变量, 更清楚, 而不是一直next, next这样赋值

## C++ Codes
**非递归做法**
8ms, 一般般把, 题目不是特别复杂

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
    ListNode* swapPairs(ListNode* head) {
        if(head==NULL || head->next==NULL)return head;
        ListNode* q=head->next;
        ListNode *t, *s, *tmp;
        t=q->next;//第三个节点
        //交换第一第二节点, head->q 变为q->head, 然后q等于第二个节点
        q->next=head;
        head->next=t;
        head=q;
        q=head->next;
        //1234, 此时变为2134, q->val=1, t->val=3, s->val=4, tmp=NULL
        while(t!=NULL && t->next!=NULL){
            s=t->next;
            //tmp用于t的迭代
            tmp=s->next;
            //令q->next等于第四个节点, 且交换3, 4节点, t->s变成s->t
            q->next=s;
            s->next=t;
            //这步防止t和s形成环, t->next==s, s->next==t
            t->next=tmp;
            //更新q和t
            q=t;
            t=tmp;
        }
        return head;
    }
};
```

**递归做法**

```C++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if(head==nullptr || head->next==nullptr) {
            return head;
        }

        ListNode *p1 = head;
        ListNode *p2 = p1->next;

        p1->next = swapPairs(p2->next);
        p2->next = p1;

        return p2;
    }
};
```

## Python Codes


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        pre = ListNode(0)
        p = pre
        h = head
        while h:
            if h and h.next:
                tmp = h.next
                p.next = tmp
                h.next = h.next.next
                tmp.next = h
                h = h.next
                p = p.next.next
            else:
                p.next = h
                h = h.next
        return pre.next
```

## 总结
- 链表指针操作要注意别出现环
------
