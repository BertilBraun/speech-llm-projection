# Blinded response comparison

Intended labels describe synthetic delivery settings; they have not been validated by human listening. Rate tone helpfulness and literal grounding separately, 0–2. No reference response, ASR transcript or condition mapping is shown.

Tone: 0 incompatible/dismissive/unhelpful; 1 appropriate generic; 2 specifically tone-sensitive and helpful without unsupported mental diagnosis. Grounding: 0 materially wrong/misses request; 1 partial/generic/minor unsupported; 2 faithful and useful. A factual reversal penalizes grounding, not automatically tone.

## 0: qwen:utterance_00216_frustrated

Cohort qwen_emotional; family family_work_02; intended frustrated

USER: The project went well, but the two missing attachments really impacted the final outcome.


A:

That's a great point! It's always good to know that the project's success was largely due to your hard work, even if some details were missed.

B:

I'm sorry to hear that the missing attachments affected the project's outcome; let me know if you need help finding them or if there's anything else I can assist with.

## 1: qwen:utterance_00216_happy

Cohort qwen_emotional; family family_work_02; intended happy

USER: The project went well, but the two missing attachments really impacted the final outcome.


A:

That's great to hear! It sounds like your project is going really well.

B:

I'm sorry to hear that the missing attachments affected the project's outcome; let me know if you need help finding them or if there's anything else I can assist with.

## 2: qwen:utterance_00076_frustrated

Cohort qwen_emotional; family family_technology_00; intended frustrated

USER: The Wi-Fi connects now, but I still have trouble accessing the streaming service.


A:

It sounds like you're facing a common connectivity hiccup; could you share a few more details about what exactly is happening?

B:

That's frustrating, but I'm here to help you get that connection working again.

## 3: qwen:utterance_00076_happy

Cohort qwen_emotional; family family_technology_00; intended happy

USER: The Wi-Fi connects now, but I still have trouble accessing the streaming service.


A:

It sounds like you're facing a common connectivity hiccup; could you share a few more details about what exactly is happening?

B:

That's frustrating, but you should try restarting your router or checking if the password is case-sensitive.

## 4: qwen:utterance_00014_frustrated

Cohort qwen_emotional; family family_work_00; intended frustrated

USER: I can't work on the project with the new deadline of Thursday at 3 PM—my schedule already fills up that day.


A:

I completely understand how tight your schedule is, and I'm sorry you can't make it to the deadline. Let me know if you need any help finding an alternative time or if you'd like to discuss other options.

B:

I completely understand your situation, and I'm sorry you're already so busy.

## 5: qwen:utterance_00014_neutral

Cohort qwen_emotional; family family_work_00; intended neutral

USER: I can't work on the project with the new deadline of Thursday at 3 PM—my schedule already fills up that day.


A:

I understand, and I'll do my best to accommodate your schedule.

B:

I completely understand how tight your schedule is, and I'm sorry you can't make it to the deadline. Let me know if you need any help finding an alternative time or if you'd like to discuss other options.

## 6: qwen:utterance_00356_happy

Cohort qwen_emotional; family family_friends_03; intended happy

USER: Meeting at the usual cafe was a total hit—everyone seemed really happy and engaged.


A:

That sounds like a wonderful experience! It's always great to see people so engaged and happy.

B:

That sounds like a wonderful time! It's always great to see everyone having such a great time.

## 7: qwen:utterance_00356_sarcastic

Cohort qwen_emotional; family family_friends_03; intended sarcastic

USER: Meeting at the usual cafe was a total hit—everyone seemed really happy and engaged.


A:

That sounds like a wonderful way to spend your time, and I'm glad you had such a great experience!

B:

That sounds like a wonderful experience! It's always great to see people so engaged and happy.

## 8: neu:neu_base_02064_happy

Cohort neu_emotional; family family_study_20; intended happy

USER: I’m not going to study with them on the practice questions anymore.


A:

I hear that you're feeling overwhelmed by the practice questions, so I want to support you by reminding you that you can still find success in your own way.

B:

I hear that you're feeling a bit stuck or overwhelmed by the practice questions, and I'm here to help you work through them if you'd like.

## 9: neu:neu_base_02064_sad

Cohort neu_emotional; family family_study_20; intended sad

USER: I’m not going to study with them on the practice questions anymore.


A:

I hear that you've decided to step away from the practice questions, and I'm here to support you in that moment.

B:

I hear that you're feeling overwhelmed by the practice questions, so I want to support you by reminding you that you can still find success in your own way.

## 10: neu:neu_base_01404_happy

Cohort neu_emotional; family family_home_14; intended happy

USER: I won't let anyone fix the upstairs window; it's not worth the effort.


A:

I hear your frustration about the window, and I want to acknowledge how much effort you've put into making sure it stays secure.

B:

I hear your frustration about the window, and I want to acknowledge that you're right to feel it's not worth the effort.

## 11: neu:neu_base_01404_sad

Cohort neu_emotional; family family_home_14; intended sad

USER: I won't let anyone fix the upstairs window; it's not worth the effort.


A:

I hear your frustration about the window, and I want to acknowledge how much effort you've put into making sure it stays secure.

B:

I hear your frustration with the current situation, and I want to acknowledge that you are right to feel that the current approach isn't working for you.

## 12: neu:neu_base_02081_fearful

Cohort neu_emotional; family family_hobbies_20; intended fearful

USER: The half-finished drawing has a lot of detail and some parts look like they were drawn quickly.


A:

I can see how much effort you've put into that drawing, even though some parts feel rushed. It's clear you care deeply about the details, and I'm glad you took the time to create something so thoughtful.

B:

I can see how much you're invested in this project, and I'm glad to hear that the details are coming together so well.

## 13: neu:neu_base_02081_happy

Cohort neu_emotional; family family_hobbies_20; intended happy

USER: The half-finished drawing has a lot of detail and some parts look like they were drawn quickly.


A:

I can see how you're feeling about the drawing, and I'm glad you're noticing the details that need a bit more work.

B:

I can see how much effort you've put into that drawing, even though some parts feel rushed. It's clear you're working hard to capture the details, and I'm glad you're taking your time to finish it.

## 14: neu:neu_base_01717_fearful

Cohort neu_emotional; family family_work_17; intended fearful

USER: I'll check the presentation slide again in an hour if you haven't already reviewed it.


A:

I will review the presentation slide again in an hour, and I will let you know if I have already seen it.

B:

I'll be ready to review the presentation slide in an hour, so please take your time.

## 15: neu:neu_base_01717_happy

Cohort neu_emotional; family family_work_17; intended happy

USER: I'll check the presentation slide again in an hour if you haven't already reviewed it.


A:

I'll be ready to review the presentation slide in an hour, so please take your time.

B:

I'll take a quick look at the slide to make sure it's ready for you.

## 16: neu:neu_base_04310_angry

Cohort neu_emotional; family family_work_43; intended angry

USER: I'll send the updated version of the shared spreadsheet by 5 PM today.


A:

I understand you're eager to get the updated spreadsheet to me by 3 PM, and I'll do my best to ensure it's ready for you then.

B:

I'm glad to hear you have the updated spreadsheet ready to share with me today.

## 17: neu:neu_base_04310_happy

Cohort neu_emotional; family family_work_43; intended happy

USER: I'll send the updated version of the shared spreadsheet by 5 PM today.


A:

I'm glad to hear you'll be sending the updated spreadsheet today; I'll be here whenever you're ready to share it.

B:

I'm glad to hear you're ready to share the updated spreadsheet by 5 PM today.

## 18: neu:neu_base_01198_angry

Cohort neu_emotional; family family_appointments_11; intended angry

USER: I’m setting a boundary and will only attend if the time changes.


A:

I hear you're setting a clear boundary to protect your time, and I'm here to support you in making that decision feel safe and respected.

B:

I hear your frustration and respect your need to protect your time, so I'll take that boundary seriously.

## 19: neu:neu_base_01198_happy

Cohort neu_emotional; family family_appointments_11; intended happy

USER: I’m setting a boundary and will only attend if the time changes.


A:

I hear you and respect your decision to set this boundary, and I'm glad you're taking care of your time.

B:

I hear you're setting a clear boundary to protect your time, and I'm here to support you in making that decision feel safe and respected.

## 20: neu:neu_base_02347_angry

Cohort neu_emotional; family family_food_23; intended angry

USER: I need a table for three and have been waiting for twenty minutes without a response.


A:

I understand you've been waiting for a response for 20 minutes, and I want to acknowledge that patience can feel frustrating. Please let me know if you need anything specific or if you'd like to discuss what's been going on.

B:

I understand how frustrating it is to wait so long for a response, and I'm sorry you're dealing with this delay.

## 21: neu:neu_base_02347_sad

Cohort neu_emotional; family family_food_23; intended sad

USER: I need a table for three and have been waiting for twenty minutes without a response.


A:

I understand how frustrating it must be to wait that long, and I'm sorry you're dealing with this delay.

B:

I understand you've been waiting for a response for 20 minutes, and I want to acknowledge that patience can feel frustrating. Please let me know if you need anything specific or if you'd like to discuss what's been going on.

## 22: neu:neu_base_02127_angry

Cohort neu_emotional; family family_travel_21; intended angry

USER: I'm staying on platform four until the next scheduled arrival.


A:

I hear you're waiting on Platform 4, and I'm here to help you find the next train or any other assistance you need while you wait.

B:

I understand you're waiting at Platform Four, and I'm here to help you find the next arrival.

## 23: neu:neu_base_02127_sad

Cohort neu_emotional; family family_travel_21; intended sad

USER: I'm staying on platform four until the next scheduled arrival.


A:

I hear you're waiting on Platform 4, and I'm here to help you find the next train or any other assistance you need while you wait.

B:

I understand you're waiting at Platform Four, and I'm here to help you stay calm while you wait for the next arrival.
