---
title: LeetCode-025-Reverse Nodes in k-Group
tags:
  - LeetCode
  - Algorithm
  - Stack
  - Recursion
  - LinkedList
  - Reverse
  - Hard
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-07-25 00:46:48
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/submissions/)   
Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.

k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.

### Examples:
**Input:** 1-2-3-4-5
**Output:** 
> k=2, 2-1-4-3-5
> k=3, 3-2-1-4-5
{% endnote %}
<!--more-->

## Solutions
- 递归, 每次改变k个节点的链接顺序, 然后令最后一个节点的next等于下一次递归的返回值, 最后返回头节点, 如果本次不满k个, 直接返回. 每次递归的内容可以是迭代或者用栈, 两种方式的时间差不多
- 迭代, 基于第24题的迭代, 每次修改完一段之后, 要将此段和前一段相连, 还要和后一段相连, 要注意很多细节
- 利用栈, 因为k个节点进栈顺序和出栈顺序正好相反


## C++ Codes
节点数据结构

```C++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */

```

递归+栈

```C++
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if(k==1 || head==NULL) return head;
        ListNode* stack[k];
        ListNode *p=head;
        //存入栈中
        for(int i=0;i<k;i++){
            if(p==NULL)return head;
            stack[i]=p;
            p=p->next;
        }
        //逆序
        for(int i=k-1;i>0;i--){
            stack[i]->next=stack[i-1];
        }
        //和后一段相连
        stack[0]->next=reverseKGroup(p, k);
        //每次递归返回头节点
        return stack[k-1]; 
    }
};
```

纯迭代, 相对递归感觉比较麻烦, 逻辑上要想更多, 但是资源肯定比递归消耗少
```C++
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if(k==1) return head;
        ListNode *front, *rear, *tmp, *p, *q, *s;
        front=head;
        for(int i=0;i<k;i++){
            if(front!=NULL) front=front->next;
            else return head;
        }
        front=rear=head;
        //front指向第一个节点, rear指向k+1个节点
        for(int i=0;i<k;i++){
            if(i==k-1) head=rear;
            rear=rear->next;
        }
        
        //pre是前一段最后一个节点
        ListNode *pre=new ListNode(0);
        while(true){
            p=front;q=front->next;s=q->next;
            while(q!=rear){
                q->next=p;
                p=q;q=s;
                //这里要注意rear可能是NULL, s->next需判断
                if(s==rear && s==NULL) break;
                else s=s->next;                
            }
            //本段和后一段相连,此时q=rear, s=rear->next, 
            front->next=rear;
            //前面一段和本段第一个节点相连 
            pre->next=p;
            pre=front;
            //更新front和rear
            front=rear;
            for(int i=0;i<k;i++) {
                //rear可能没到目标就成了NULL
                if(rear==NULL)return head;
                rear=rear->next;
            }
        }
        return head;  
    }
};
```

## Python Codes
递归+栈

```python
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if k==1 or head==None:
            return head
        stack=[]
        p=head;
        #存入栈中
        for i in range(k):
            if p==None:
                return head
            stack.append(p);
            p=p.next
        
        #逆序
        for i in range(k-1, 0, -1):
            stack[i].next=stack[i-1]
        #和后一段相连, python递归要self.reverseKGroup()
        stack[0].next=self.reverseKGroup(p, k)
        #每次递归返回头节点
        return stack[k-1] 
```

迭代

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if k==1:
            return head
        front=head
        for i in range(k):
            if front!=None:
                front=front.next
            else:
              return head    
        
        front=rear=head
        #front指向第一个节点, rear指向k+1个节点
        for i in range(k):
            if i==k-1:
                head=rear
            rear=rear.next 
            
        #pre是前一段最后一个节点
        pre=ListNode(0);
        while True:
            p=front
            q=front.next
            s=q.next
            while q!=rear:
                q.next=p
                p=q
                q=s
                #这里要注意rear可能是NULL, s->next需判断
                if s==rear and s==None:
                    break
                else:
                    s=s.next                     
            #本段和后一段相连,此时q=rear, s=rear->next, 
            front.next=rear
            #前面一段和本段第一个节点相连 
            pre.next=p
            pre=front
            #新front和rear
            front=rear;
            for i in range(k):
                #rear可能没到目标就成了NULL
                if rear==None:
                    return head
                rear=rear.next
        return head;  
```

## 总结
- 逆序问题可利用栈的特性
- 发现问题可以分为多个子问题, 利用递归
- 迭代的过程注意避免空指针和环

-------
Python是真的不太熟悉啦
