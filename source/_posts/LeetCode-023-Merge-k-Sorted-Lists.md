---
title: LeetCode-023-Merge k Sorted Lists
tags:
  - LeetCode
  - List
  - LinkedList
  - PriorityQueue
  - Merge
  - Hard
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-05-23 13:10:02
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/merge-k-sorted-lists/)   
Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

### Examples:
**Input:**
> [
> 　　1->4->5,
> 　　1->3->4,
> 　　2->6
> ]
**Output:** 1->1->2->3->4->4->5->6

{% endnote %}
<!--more-->

## Solutions
- 相对上一题难度加大了一点，但是如果按照上一题的思路也是可以做的，就是时间复杂度会有点高
- 这里介绍两种方法，第一种就是类似上一题的解法，每次循环找到所有链表中最小的头结点，然后改变指针指向，并不新建节点，中间有个坑是测试用例中会有空的链表，要注意处理下。每次加一个节点，加的时候遍历所有链表，所以时间复杂度是$$ O(n \times k) $$,n是节点总数，k是链表数
- 第二种方法是使用优先级队列，先将所有链表加入队列中，每次从队列中找到最小的节点，然后也是找n个节点，相对上一种方法是在$$ O(n \times k) $$的k这里进行了优化，因为优先级队列存取是log()级别，因此时间复杂度是$$ O(n \times log(k)) $$
- 还想到一种方法是迭代，用上一题的方法进行两两合并，没有实现这种方法，看了题解有分治法，使用递归进两两合并


## C++ Codes

### 方法一
就地改变链表指针指向，时间复杂度$$ O(n \times k) $$

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
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if(lists.size()==0)return NULL;
        ListNode *res = new ListNode(0);
        ListNode *pre = res;
        int min = 0;

        while(lists.size()!=0){ 
            //确保lists[0]非NULL
            while(lists.size()>0 && lists[min]==NULL){
                lists.erase(lists.begin()+min);
                min = 0;
            }
            if(lists.size()==0) return res->next;
   
            for(int i=0;i<lists.size();i++){
                if(lists[i]==NULL) 
                    continue;
                if(lists[i]->val<lists[min]->val)
                    min = i;
            }
            pre->next = lists[min];
            pre = pre->next;
            lists[min] = lists[min]->next;
        }
        
        return res->next;
    }
};
```

### 方法二
使用优先级队列进行每轮插入最小的节点，时间复杂度为$$ O(n \times log(k)) $$

这里使用最小优先队列，最小的值优先级最大

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
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        //定义优先级队列比较结构
        struct cmp{
            bool operator ()(ListNode* a, ListNode* b){
                return a->val > b->val;
            }
        };
        ListNode *res = new ListNode(0);
        ListNode *p = res;
        
        //建立优先级队列，这里是最小优先队列，最小的值优先级最大
        priority_queue<ListNode*, vector<ListNode*>, cmp> pq;
        for(ListNode* list : lists){
            if(list!=NULL) pq.push(list);
        }
        
        while(pq.size()>0){
            p->next = pq.top();//获取优先级最高的元素，即数字最小的
            pq.pop();//删除队首元素
            p = p->next;
            if(p->next != NULL) pq.push(p->next);
        }
        return res->next;
    }
};
```

###方法三
分治法，使用递归两两合并，最终的链表等于合并好的前半部分加上合并好的后半部分，对前半部分在进行半部分和半部分的合并。

下面贴一下题解的代码，Java

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
   public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        return merge(lists, 0, lists.length - 1);
    }

    private ListNode merge(ListNode[] lists, int left, int right) {
        if (left == right) return lists[left];
        int mid = left + (right - left) / 2;
        ListNode l1 = merge(lists, left, mid);
        ListNode l2 = merge(lists, mid + 1, right);
        return mergeTwoLists(l1, l2);
    }

    private ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1,l2.next);
            return l2;
        }
    }
}
```


## Python Codes
这里使用方法二

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        import heapq
        dummy = ListNode(0)
        p = dummy
        head = []
        for i in range(len(lists)):
            if lists[i] :
                heapq.heappush(head, (lists[i].val, i))
                lists[i] = lists[i].next
        while head:
            val, idx = heapq.heappop(head)
            p.next = ListNode(val)
            p = p.next
            if lists[idx]:
                heapq.heappush(head, (lists[idx].val, idx))
                lists[idx] = lists[idx].next
        return dummy.next
```

## 总结
- 合并多个链表相对上一题就是链表数变多之后的处理，可以用优先级队列加快节点的查找，也可以使用分治法，合并症各部分就是先合并前半部分和后半部分，再将这两个部分合在一起 


------
