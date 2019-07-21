---
title: LeetCode-002-Add Two Numbers
comments: true
mathjax: false
date: 2019-04-09 00:39:36
tags: [LeetCode, Linked List, Large Number, Math, Medium]
categories: [LeetCode]
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/add-two-numbers/)
You are given two **non-empty** linked lists representing two non-negative integers. 
The digits are stored in **reverse order** and each of their nodes contain a single digit. 
Add the two numbers and return it as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.

### Example:
> Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
> Output: 7 -> 0 -> 8
> Explanation: 342 + 465 = 807.
{% endnote %}
<!--more-->

## Solutions
因为以前做过差不多的题目，而且感觉也不难，设置一个进位标志，然后一个个往后加。

犯了三个小错误
- 算当前node的值时居然没把carry一起加起来算
- 算下次的carry时没有把上次的carry加上
- 忘了最后一次还可能有进位，要新建一个node

还纠结好久要不要头结点，刚开始以为题目不可以用头结点，毕竟Example上没有，写到后面还是出错了，看了题解换了有头结点的版本就好了。

## C++ Codes

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
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int carry = 0;
        ListNode *res = new ListNode(0);
        ListNode *tmp = res;
        ListNode *p = l1;
        ListNode *q = l2;

        while(p!=NULL && q!=NULL){
            tmp->next = new ListNode((p->val + q->val+carry)%10);//把+carry放在了%10的后面
            carry = (p->val + q->val+carry)/10;//居然漏了+carry

            tmp = tmp->next;

            p=p->next;
            q=q->next;
        }

        while(p!=NULL){
            tmp->next = new ListNode((p->val + carry)%10);
            carry = (p->val + carry)/10;
            tmp = tmp->next;
            p = p->next;
        }

        while(q!=NULL){
            tmp->next = new ListNode((q->val + carry)%10);
            carry = (q->val + carry)/10;
            tmp = tmp->next;
            q = q->next;
        }

        if(carry>0) tmp->next = new ListNode(carry);//漏了最后的进位
        return res->next;
    }
};
```
{% note default %}
我这里在最后用了两个while来判断是否结束，题解上面是在第一个while里面对p和q进行了处理：
　　只要到了NULL，就让它的值为0，时间复杂度上来说是一样的，都是$$ \max{l1的节点数，l2的节点数} $$，简单理解为$$ O(n) $$
{% endnote %}

## Python Codes

一共两种，不是自己写的，不是很熟悉Python的链表，多看看学习下。

第一中的sum写法不是很懂...绝望，第二种比较好理解

```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        res = ListNode(0)
        tmp = res
        left = 0

        while l1 or l2 or left:
            left, right = divmod(sum(l and l.val or 0 for l in (l1, l2)) + left, 10)
            tmp.next = ListNode(right)
            tmp = tmp.next
            l1 = l1 and l1.next
            l2 = l2 and l2.next

        return res.next
```

```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(0)
        curr = head
        carry, total = 0, 0

        while(l1 or l2):
            a = l1.val if l1 else 0
            b = l2.val if l2 else 0
            total = a + b + carry
            carry = total // 10

            if l1 or l2:
                curr.next = ListNode(total % 10)

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
            curr = curr.next

        if carry>0:
            curr.next = ListNode(carry)

        return head.next
```

## 总结
感觉自己解题速度有点慢，而且粗心犯小错误。

感觉是练得太少，以后加油！

-----------
