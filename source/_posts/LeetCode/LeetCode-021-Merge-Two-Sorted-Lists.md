---
title: LeetCode-021-Merge Two Sorted Lists
tags:
  - LeetCode
  - Algorithm
  - LinkedList
  - List
  - Merge
  - Easy
categories:
  - LeetCode
urlname: leetcode-merge-two-sorted-lists
comments: true
mathjax: false
date: 2019-05-23 00:28:14
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/merge-two-sorted-lists/)   
Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

简单说就是合并两个有序链表

### Examples:
**Input:**1->2->4, 1->3->4
**Output:**1->1->2->3->4->4

{% endnote %}
<!--more-->

## Solutions
- 第一种方法是新建一个链表，每次创建新的节点，不影响原始链表。在两个链表都非空的是比较结点值大小，将小的值作为新节点的值插入结果链表中，并移动指针。
- 第二种方法也挺简单，只是单纯的改变指针，就地合并，但是会影响原始链表。
- Python使用的方法，递归，也是就地的，前面的较小的节点加上后面所有的排好序的节点就是要的结果


## C++ Codes

### 方法一
创建一个头结点，依次比较l1和l2的节点，逐个插入到新链表中，插入过程是创建了新的节点

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
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* res = new ListNode(0);
        ListNode *p = res;
        ListNode *p1 = l1;
        ListNode *p2 = l2;
        while(p1!=NULL && p2!=NULL){
            if(p1->val > p2->val){
                p->next = new ListNode(p2->val);
                p2 = p2->next;
            }
            else{
                p->next = new ListNode(p1->val);
                p1 = p1->next;
            }
            p = p->next;
        }
        while(p1!=NULL){
            p->next = new ListNode(p1->val);
            p = p->next;
            p1 = p1->next;
        }
        while(p2!=NULL){
            p->next = new ListNode(p2->val);
            p = p->next;
            p2 = p2->next;
        }
        return res->next;
    }
};
```

### 方法二
这种方法也可以像方法一那样，创建一个头结点，然后比较l1和l2，依次改变两个链表的指针

我用了另一种就地方法，其中一个链表不变，另一个链表依次找到合适位置插入这个这个链表

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
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        //p1: 结果链表头结点，指向较小的
        //p2: 另一个链表的头结点
        if(l1==NULL)return l2;
        if(l2==NULL)return l1;
        ListNode *p1, *p2;
        if(l1->val < l2->val){
            p1 = l1; p2 = l2;
        }else{
            p1 = l2; p2 = l1;
        }
        ListNode *prefix=p1, *res = p1;
        p1 = p1->next;
        while(p2!=NULL){
            //找到p2指向的节点第一次小于p1节点的位置，前插
            while(p1!=NULL && p2->val > p1->val){
                prefix = prefix->next;
                p1 = p1->next;
            }
            //p1到结尾
            if(p1==NULL){
                prefix->next = p2;
                return res;
            }
            else{
                prefix->next = p2;
                p2 = p2->next;
                prefix->next->next = p1;
                prefix = prefix->next;
            }
            
        }
        return res;
    }
};
```

## Python Codes
递归的方法，每次保证递归返回的是较小的节点

这里用了交换l1和l2指针的方法，事实证明....不交换直接判断哪个小来调用递归时间少一点...佛了，交换是64ms，不交换是48ms...

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1: return l2
        if not l2: return l1
        if l1.val>l2.val: l1, l2 = l2, l1
        l1.next = self.mergeTwoLists(l1.next,l2)
        return l1
```

## 总结
- 链表的问题这里总结是两种方法，一种是迭代，一种是递归，迭代过程中要注意节点指针不要混乱了。


------
