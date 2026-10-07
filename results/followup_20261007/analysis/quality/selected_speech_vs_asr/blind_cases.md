# Blinded response comparison

Intended labels describe synthetic delivery settings; they have not been validated by human listening. Rate tone helpfulness and literal grounding separately, 0–2. No reference response, ASR transcript or condition mapping is shown.

Tone: 0 incompatible/dismissive/unhelpful; 1 appropriate generic; 2 specifically tone-sensitive and helpful without unsupported mental diagnosis. Grounding: 0 materially wrong/misses request; 1 partial/generic/minor unsupported; 2 faithful and useful. A factual reversal penalizes grounding, not automatically tone.

## 0: qwen:utterance_00011_happy

Cohort qwen_emotional; family family_work_00; intended happy

USER: Great news, the deadline has been moved to Thursday at 3 PM, which gives us more time to finalize the report.


A:

That's wonderful to hear, and I'm glad the deadline has been pushed to Thursday!

B:

That's wonderful news, and I'm glad the deadline has been extended to give you more time to finalize your report.

## 1: qwen:utterance_00011_sad

Cohort qwen_emotional; family family_work_00; intended sad

USER: Great news, the deadline has been moved to Thursday at 3 PM, which gives us more time to finalize the report.


A:

I'm glad to hear the deadline has been moved to Thursday, and I hope the new schedule helps you get through the rest of the week.

B:

That's wonderful to hear, and I'm glad the deadline has been pushed to Thursday!

## 2: qwen:utterance_00012_neutral

Cohort qwen_emotional; family family_work_00; intended neutral

USER: Wait, you said the deadline is Thursday at 3 PM? That doesn't match what I was told earlier.


A:

I apologize for the confusion; I will ensure the deadline is clearly communicated as Thursday at 3 PM moving forward.

B:

I apologize for the confusion; I must have misremembered the details. Could you please clarify the correct deadline so we can get this straightened out immediately?

## 3: qwen:utterance_00012_sarcastic

Cohort qwen_emotional; family family_work_00; intended sarcastic

USER: Wait, you said the deadline is Thursday at 3 PM? That doesn't match what I was told earlier.


A:

I apologize for the confusion; I must have misremembered the details. Could you please clarify the correct deadline so we can get this straightened out immediately?

B:

I apologize for the confusion; I will ensure the deadline is clearly stated as Thursday at 3 PM going forward.

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

## 6: qwen:utterance_00015_happy

Cohort qwen_emotional; family family_work_00; intended happy

USER: Thanks for adjusting the deadline to Thursday at 3 PM, that really helps with our timeline.


A:

I'm glad the deadline adjustment works out for your schedule!

B:

I'm glad the deadline adjustment works out for you, and I'll make sure to keep the team updated on the new schedule.

## 7: qwen:utterance_00015_sarcastic

Cohort qwen_emotional; family family_work_00; intended sarcastic

USER: Thanks for adjusting the deadline to Thursday at 3 PM, that really helps with our timeline.


A:

I'm glad the deadline adjustment works out for you, and I'll make sure to keep the team updated on the new schedule.

B:

I'm glad the deadline adjustment works out for your schedule!

## 8: qwen:utterance_00071_happy

Cohort qwen_emotional; family family_technology_00; intended happy

USER: The Wi-Fi password is now working, though it took a while to get right.


A:

That's great news! Glad to hear your Wi-Fi is finally up and running.

B:

That's great news! It's always exciting to finally get the Wi-Fi working after all that hassle.

## 9: qwen:utterance_00071_sad

Cohort qwen_emotional; family family_technology_00; intended sad

USER: The Wi-Fi password is now working, though it took a while to get right.


A:

I'm sorry to hear that, but I'm here to help you get that connection working again.

B:

That's great news! It sounds like you finally cracked the code after some trial and error.

## 10: qwen:utterance_00076_frustrated

Cohort qwen_emotional; family family_technology_00; intended frustrated

USER: The Wi-Fi connects now, but I still have trouble accessing the streaming service.


A:

It sounds like you're facing a common connectivity hiccup; could you share a few more details about what exactly is happening?

B:

That's frustrating, but I'm here to help you get that connection working again.

## 11: qwen:utterance_00076_happy

Cohort qwen_emotional; family family_technology_00; intended happy

USER: The Wi-Fi connects now, but I still have trouble accessing the streaming service.


A:

It sounds like you're facing a common connectivity hiccup; could you share a few more details about what exactly is happening?

B:

That's frustrating, but you should try restarting your router or checking if the password is case-sensitive.

## 12: qwen:utterance_00078_neutral

Cohort qwen_emotional; family family_technology_00; intended neutral

USER: I’ll only ask for the Wi-Fi password if it’s absolutely necessary.


A:

I understand your concern about privacy, but I can't ask for a password without it being necessary.

B:

I completely understand your preference for privacy, and I'll certainly only share the password if you absolutely need it.

## 13: qwen:utterance_00078_sad

Cohort qwen_emotional; family family_technology_00; intended sad

USER: I’ll only ask for the Wi-Fi password if it’s absolutely necessary.


A:

I completely understand your preference for privacy, and I'll certainly only share the password if you absolutely need it.

B:

I understand your concern about privacy, but I can't ask for your password without your explicit permission.

## 14: qwen:utterance_00216_frustrated

Cohort qwen_emotional; family family_work_02; intended frustrated

USER: The project went well, but the two missing attachments really impacted the final outcome.


A:

That's a great point! It's always good to know that the project's success was largely due to your hard work, even if some details were missed.

B:

I'm sorry to hear that the missing attachments affected the project's outcome; let me know if you need help finding them or if there's anything else I can assist with.

## 15: qwen:utterance_00216_happy

Cohort qwen_emotional; family family_work_02; intended happy

USER: The project went well, but the two missing attachments really impacted the final outcome.


A:

That's great to hear! It sounds like your project is going really well.

B:

I'm sorry to hear that the missing attachments affected the project's outcome; let me know if you need help finding them or if there's anything else I can assist with.

## 16: qwen:utterance_00356_happy

Cohort qwen_emotional; family family_friends_03; intended happy

USER: Meeting at the usual cafe was a total hit—everyone seemed really happy and engaged.


A:

That sounds like a wonderful experience! It's always great to see people so engaged and happy.

B:

That sounds like a wonderful time! It's always great to see everyone having such a great time.

## 17: qwen:utterance_00356_sarcastic

Cohort qwen_emotional; family family_friends_03; intended sarcastic

USER: Meeting at the usual cafe was a total hit—everyone seemed really happy and engaged.


A:

That sounds like a wonderful way to spend your time, and I'm glad you had such a great experience!

B:

That sounds like a wonderful experience! It's always great to see people so engaged and happy.

## 18: qwen:utterance_00359_frustrated

Cohort qwen_emotional; family family_friends_03; intended frustrated

USER: Let's meet at the usual cafe, but only if it's not too late—my schedule gets busy after 7 PM.


A:

I completely understand your frustration with the late schedule; let's try to find a time that works for you both.

B:

I'd love to meet then, but I'll need to check my calendar to see if we can make it before 7pm.

## 19: qwen:utterance_00359_neutral

Cohort qwen_emotional; family family_friends_03; intended neutral

USER: Let's meet at the usual cafe, but only if it's not too late—my schedule gets busy after 7 PM.


A:

I'll make sure to check the schedule and find a time that works for you.

B:

I'll make sure to keep my schedule flexible so we can meet at the usual cafe before it gets too late.

## 20: qwen:utterance_00971_frustrated

Cohort qwen_emotional; family family_technology_09; intended frustrated

USER: The shared calendar shows the meeting at 3 PM, but I didn't see it in my notifications.


A:

It sounds like you're feeling a bit frustrated about the missing notification, but I'm here to help you figure out why that happened.

B:

It seems you might have missed the notification, so let's double-check your inbox or calendar to make sure you don't miss any important details.

## 21: qwen:utterance_00971_neutral

Cohort qwen_emotional; family family_technology_09; intended neutral

USER: The shared calendar shows the meeting at 3 PM, but I didn't see it in my notifications.


A:

It sounds like you might have missed the meeting, so you'll need to check your calendar again to make sure you don't miss anything else.

B:

It seems you might have missed the notification, so let's double-check your inbox or calendar to make sure you don't miss any important details.

## 22: qwen:utterance_00975_frustrated

Cohort qwen_emotional; family family_technology_09; intended frustrated

USER: Thanks for fixing the shared calendar—I finally see the event I missed yesterday.


A:

That's wonderful to hear! I'm glad you were able to catch up on the event you missed.

B:

That's wonderful news! I'm glad the calendar is finally working for you.

## 23: qwen:utterance_00975_happy

Cohort qwen_emotional; family family_technology_09; intended happy

USER: Thanks for fixing the shared calendar—I finally see the event I missed yesterday.


A:

That's wonderful to hear! I'm glad you were able to catch up on the event you missed.

B:

That's wonderful news! I'm glad the calendar is finally working for you.

## 24: neu:neu_base_00011_angry

Cohort neu_emotional; family family_work_00; intended angry

USER: The revised deadline for the project was pushed to next week, which affects the timeline.


A:

I understand the pressure of the revised deadline, and I'm sorry to hear that the timeline has shifted.

B:

I hear that the deadline has been moved to next week, and I want to acknowledge how that shift impacts your current schedule.

## 25: neu:neu_base_00011_sad

Cohort neu_emotional; family family_work_00; intended sad

USER: The revised deadline for the project was pushed to next week, which affects the timeline.


A:

I understand that the revised deadline has shifted the timeline, and I'm here to help you navigate this change.

B:

I hear that the deadline has been moved to next week, and I want to acknowledge how that shift impacts your current schedule.

## 26: neu:neu_base_00016_happy

Cohort neu_emotional; family family_work_00; intended happy

USER: The project will be completed by Thursday, but that's later than expected.


A:

I hear that the project is still on track for Thursday, but I understand that the timeline has slipped a bit. Let's take a moment to acknowledge that this delay is unexpected and see if we can figure out how to get back on schedule.

B:

I understand that the project is falling behind its expected timeline, and I'm here to help you navigate this delay.

## 27: neu:neu_base_00016_sad

Cohort neu_emotional; family family_work_00; intended sad

USER: The project will be completed by Thursday, but that's later than expected.


A:

I hear that the project is still on track for Thursday, but I understand that the timeline has slipped a bit. Let's take a moment to acknowledge that this delay is unexpected and see if we can figure out how to get back on schedule.

B:

I hear your concern about the delay, and I want to acknowledge that this is a frustrating situation for you.

## 28: neu:neu_base_00073_fearful

Cohort neu_emotional; family family_technology_00; intended fearful

USER: The Wi-Fi password I was given is incorrect, please provide the right one.


A:

I understand you're frustrated that the Wi-Fi password you received doesn't work, and I'm here to help you find the correct one.

B:

I understand you are worried about the wrong password being sent, and I will ensure the correct one is provided immediately.

## 29: neu:neu_base_00073_happy

Cohort neu_emotional; family family_technology_00; intended happy

USER: The Wi-Fi password I was given is incorrect, please provide the right one.


A:

I'm glad to hear you've found the correct password, and I'm happy to help you get it into place.

B:

I understand you're frustrated that the Wi-Fi password you received doesn't work, and I'm here to help you find the correct one.

## 30: neu:neu_base_00215_angry

Cohort neu_emotional; family family_work_02; intended angry

USER: Thanks for sending the updates, though two attachments are still missing from the original file.


A:

I understand that the missing attachments are frustrating, and I appreciate you letting me know so I can follow up to get them for you.

B:

I'm glad you received the updates, but I'm happy to help you locate the two missing attachments so we can get everything back on track.

## 31: neu:neu_base_00215_sad

Cohort neu_emotional; family family_work_02; intended sad

USER: Thanks for sending the updates, though two attachments are still missing from the original file.


A:

I understand that the missing attachments are still a concern, and I will make sure to check again to ensure they are received.

B:

I'm glad you received the updates, but I'm happy to help you locate the two missing attachments so we can get everything back on track.

## 32: neu:neu_base_00352_happy

Cohort neu_emotional; family family_friends_03; intended happy

USER: The usual cafe is where we usually meet, but I'm not sure what time you're thinking.


A:

I'm glad you're thinking of the usual spot, as it's a place we've shared many moments. Let me know what time works best for you, and I'll make sure to be there.

B:

I'm glad you're thinking about meeting up, and I'd love to know what time works best for you.

## 33: neu:neu_base_00352_sad

Cohort neu_emotional; family family_friends_03; intended sad

USER: The usual cafe is where we usually meet, but I'm not sure what time you're thinking.


A:

I understand that the usual spot feels a bit uncertain, and I'm here to help you figure out what time works best for you.

B:

I'm glad you're thinking of the usual spot, as it's a place we've shared many moments. Let me know what time works best for you, and I'll make sure to be there.

## 34: neu:neu_base_00358_angry

Cohort neu_emotional; family family_friends_03; intended angry

USER: I'd like to meet at the usual cafe, but only if it doesn't conflict with my schedule.


A:

I'd be happy to meet you at the usual cafe, but I'll need to check your schedule first to make sure there's no conflict.

B:

I'm glad you're thinking about meeting up, and I'm happy to make sure your schedule works out for you.

## 35: neu:neu_base_00358_happy

Cohort neu_emotional; family family_friends_03; intended happy

USER: I'd like to meet at the usual cafe, but only if it doesn't conflict with my schedule.


A:

I'd be happy to meet you at the usual cafe, but I'll need to check your schedule first to make sure we don't have a conflict.

B:

I'm glad you're thinking of meeting up, and I'm happy to make sure your schedule works out for you.

## 36: neu:neu_base_00970_angry

Cohort neu_emotional; family family_technology_09; intended angry

USER: I need help resetting the smart thermostat because the schedule isn't updating correctly.


A:

I understand you're frustrated that the schedule isn't updating, and I'm here to help you get that fixed.

B:

I'm here to help you get that schedule working smoothly, and I'm ready to walk through the steps with you.

## 37: neu:neu_base_00970_happy

Cohort neu_emotional; family family_technology_09; intended happy

USER: I need help resetting the smart thermostat because the schedule isn't updating correctly.


A:

I'm here to help you get that schedule working smoothly, and I'm ready to walk through the steps with you.

B:

I'm glad you could get that schedule updated, and I'm here to help if you need any more details.

## 38: neu:neu_base_01094_angry

Cohort neu_emotional; family family_appointments_10; intended angry

USER: I can’t go to the morning slot; I have a prior commitment at noon.


A:

I hear that you have a prior commitment at noon, and I understand you can't make it to the morning slot.

B:

I understand that your prior commitment makes the morning slot unavailable, and I'm sorry you're facing this conflict.

## 39: neu:neu_base_01094_happy

Cohort neu_emotional; family family_appointments_10; intended happy

USER: I can’t go to the morning slot; I have a prior commitment at noon.


A:

I hear that you have a prior commitment at noon, and I understand you can't make it to the morning slot.

B:

I understand that your morning slot is already booked, and I'm sorry you can't make it.

## 40: neu:neu_base_01095_angry

Cohort neu_emotional; family family_appointments_10; intended angry

USER: Thank you for adjusting the morning slot to later this week.


A:

I'm glad the morning slot has been moved to later this week, and I hope it helps you get some much-needed rest.

B:

I appreciate you taking the time to make that adjustment for the morning slot.

## 41: neu:neu_base_01095_sad

Cohort neu_emotional; family family_appointments_10; intended sad

USER: Thank you for adjusting the morning slot to later this week.


A:

I'm glad the morning slot has been moved to later this week, and I hope it helps you get some much-needed rest.

B:

I appreciate you making that adjustment for the morning slot later this week.

## 42: neu:neu_base_01196_happy

Cohort neu_emotional; family family_appointments_11; intended happy

USER: I checked the email and the new time is listed correctly.


A:

It sounds like you're relieved to have confirmed the new time, and I'm glad the email is accurate.

B:

It sounds like you've found the correct time, and I'm glad to hear that the email reflects the updated schedule accurately.

## 43: neu:neu_base_01196_sad

Cohort neu_emotional; family family_appointments_11; intended sad

USER: I checked the email and the new time is listed correctly.


A:

It sounds like you've found the correct time, and I'm glad to hear that the email reflects the updated schedule accurately.

B:

I'm glad to hear the email is correct and that you've verified the new time.

## 44: neu:neu_base_01198_angry

Cohort neu_emotional; family family_appointments_11; intended angry

USER: I’m setting a boundary and will only attend if the time changes.


A:

I hear you're setting a clear boundary to protect your time, and I'm here to support you in making that decision feel safe and respected.

B:

I hear your frustration and respect your need to protect your time, so I'll take that boundary seriously.

## 45: neu:neu_base_01198_happy

Cohort neu_emotional; family family_appointments_11; intended happy

USER: I’m setting a boundary and will only attend if the time changes.


A:

I hear you and respect your decision to set this boundary, and I'm glad you're taking care of your time.

B:

I hear you're setting a clear boundary to protect your time, and I'm here to support you in making that decision feel safe and respected.

## 46: neu:neu_base_01404_happy

Cohort neu_emotional; family family_home_14; intended happy

USER: I won't let anyone fix the upstairs window; it's not worth the effort.


A:

I hear your frustration about the window, and I want to acknowledge how much effort you've put into making sure it stays secure.

B:

I hear your frustration about the window, and I want to acknowledge that you're right to feel it's not worth the effort.

## 47: neu:neu_base_01404_sad

Cohort neu_emotional; family family_home_14; intended sad

USER: I won't let anyone fix the upstairs window; it's not worth the effort.


A:

I hear your frustration about the window, and I want to acknowledge how much effort you've put into making sure it stays secure.

B:

I hear your frustration with the current situation, and I want to acknowledge that you are right to feel that the current approach isn't working for you.

## 48: neu:neu_base_01710_angry

Cohort neu_emotional; family family_work_17; intended angry

USER: I need your input on the presentation slide because the data seems off and the formatting is inconsistent.


A:

I understand your frustration with the inconsistent data and formatting, and I'm here to help you fix those issues.

B:

I'm ready to help you review the slide, but I can't see the visual content or the specific data you're referring to. Please describe the issues you're seeing, and I'll offer specific feedback on the formatting and the numbers.

## 49: neu:neu_base_01710_happy

Cohort neu_emotional; family family_work_17; intended happy

USER: I need your input on the presentation slide because the data seems off and the formatting is inconsistent.


A:

I'm ready to help you review the slide, but I can't see the visual content or the specific data you're referring to. Please describe the issues you're seeing, and I'll offer specific feedback on the formatting and the numbers.

B:

I'd be happy to review the slide and help you fix the formatting issues.

## 50: neu:neu_base_01716_happy

Cohort neu_emotional; family family_work_17; intended happy

USER: The presentation slide shows mixed results, some good, some not meeting the original goals.


A:

I can see how much you're invested in this presentation, and I'm glad to hear that the mixed results are being framed as a starting point for better goals.

B:

I hear that you're feeling a bit uncertain about the outcome, which is a common reaction when results don't perfectly match our initial expectations. It's okay to feel that way, and I'm here to help you navigate this mixed picture.

## 51: neu:neu_base_01716_sad

Cohort neu_emotional; family family_work_17; intended sad

USER: The presentation slide shows mixed results, some good, some not meeting the original goals.


A:

I hear your concern about the mixed results, and I want to acknowledge that meeting the original goals was a significant challenge.

B:

I hear that you're feeling a bit uncertain about the outcome, which is a common reaction when results don't perfectly match our initial expectations. It's okay to feel that way, and I'm here to help you navigate this mixed picture.

## 52: neu:neu_base_01717_fearful

Cohort neu_emotional; family family_work_17; intended fearful

USER: I'll check the presentation slide again in an hour if you haven't already reviewed it.


A:

I will review the presentation slide again in an hour, and I will let you know if I have already seen it.

B:

I'll be ready to review the presentation slide in an hour, so please take your time.

## 53: neu:neu_base_01717_happy

Cohort neu_emotional; family family_work_17; intended happy

USER: I'll check the presentation slide again in an hour if you haven't already reviewed it.


A:

I'll be ready to review the presentation slide in an hour, so please take your time.

B:

I'll take a quick look at the slide to make sure it's ready for you.

## 54: neu:neu_base_02064_happy

Cohort neu_emotional; family family_study_20; intended happy

USER: I’m not going to study with them on the practice questions anymore.


A:

I hear that you're feeling overwhelmed by the practice questions, so I want to support you by reminding you that you can still find success in your own way.

B:

I hear that you're feeling a bit stuck or overwhelmed by the practice questions, and I'm here to help you work through them if you'd like.

## 55: neu:neu_base_02064_sad

Cohort neu_emotional; family family_study_20; intended sad

USER: I’m not going to study with them on the practice questions anymore.


A:

I hear that you've decided to step away from the practice questions, and I'm here to support you in that moment.

B:

I hear that you're feeling overwhelmed by the practice questions, so I want to support you by reminding you that you can still find success in your own way.

## 56: neu:neu_base_02081_fearful

Cohort neu_emotional; family family_hobbies_20; intended fearful

USER: The half-finished drawing has a lot of detail and some parts look like they were drawn quickly.


A:

I can see how much effort you've put into that drawing, even though some parts feel rushed. It's clear you care deeply about the details, and I'm glad you took the time to create something so thoughtful.

B:

I can see how much you're invested in this project, and I'm glad to hear that the details are coming together so well.

## 57: neu:neu_base_02081_happy

Cohort neu_emotional; family family_hobbies_20; intended happy

USER: The half-finished drawing has a lot of detail and some parts look like they were drawn quickly.


A:

I can see how you're feeling about the drawing, and I'm glad you're noticing the details that need a bit more work.

B:

I can see how much effort you've put into that drawing, even though some parts feel rushed. It's clear you're working hard to capture the details, and I'm glad you're taking your time to finish it.

## 58: neu:neu_base_02089_fearful

Cohort neu_emotional; family family_hobbies_20; intended fearful

USER: I expect the half-finished drawing to be completed by Friday so I can review it then.


A:

I understand your urgency and the importance of having the drawing ready by Friday, so I will do my best to ensure it is completed on time.

B:

I understand you're eager to have that drawing finished by Friday so you can review it, and I'm here to help you get it done.

## 59: neu:neu_base_02089_happy

Cohort neu_emotional; family family_hobbies_20; intended happy

USER: I expect the half-finished drawing to be completed by Friday so I can review it then.


A:

I understand you're eager to have that drawing finished by Friday so you can review it, and I'm here to help you get it done.

B:

I understand you're looking forward to reviewing the drawing by Friday, and I'll do my best to get it done on time.

## 60: neu:neu_base_02124_happy

Cohort neu_emotional; family family_travel_21; intended happy

USER: I'll stay on platform four until the next train, but I don't need to wait longer than necessary.


A:

I hear your desire to move forward, so I'll keep an eye on the platform for you while you make your way to the next train.

B:

I understand you're ready to move forward, but I'm glad you're taking the time to wait for the next train.

## 61: neu:neu_base_02124_sad

Cohort neu_emotional; family family_travel_21; intended sad

USER: I'll stay on platform four until the next train, but I don't need to wait longer than necessary.


A:

I hear your desire to move forward, so I'll keep an eye on the platform for you while you make your way to the next train.

B:

I understand that waiting for the next train is taking longer than you'd like, and I'm here to support you while you wait.

## 62: neu:neu_base_02125_fearful

Cohort neu_emotional; family family_travel_21; intended fearful

USER: Thanks for confirming platform four is where the train will come in.


A:

I'm glad to hear the train is confirmed for four, and I'll keep that in mind for our next steps.

B:

I understand your concern about the train's arrival, and I'm glad to hear you're feeling reassured by the confirmation of Platform Four.

## 63: neu:neu_base_02125_happy

Cohort neu_emotional; family family_travel_21; intended happy

USER: Thanks for confirming platform four is where the train will come in.


A:

I'm glad to hear the train is confirmed for four, and I'll keep that in mind for our next steps.

B:

That's wonderful to hear! I'm glad the train is on its way to platform four.

## 64: neu:neu_base_02127_angry

Cohort neu_emotional; family family_travel_21; intended angry

USER: I'm staying on platform four until the next scheduled arrival.


A:

I hear you're waiting on Platform 4, and I'm here to help you find the next train or any other assistance you need while you wait.

B:

I understand you're waiting at Platform Four, and I'm here to help you find the next arrival.

## 65: neu:neu_base_02127_sad

Cohort neu_emotional; family family_travel_21; intended sad

USER: I'm staying on platform four until the next scheduled arrival.


A:

I hear you're waiting on Platform 4, and I'm here to help you find the next train or any other assistance you need while you wait.

B:

I understand you're waiting at Platform Four, and I'm here to help you stay calm while you wait for the next arrival.

## 66: neu:neu_base_02344_happy

Cohort neu_emotional; family family_food_23; intended happy

USER: I’ll pass on the table for three and go with a couple of seats instead.


A:

I hear you're ready to move forward with a different seating arrangement, and I'm glad you're comfortable making that choice.

B:

I'm glad you're making a choice to sit with your friends instead of sharing the table.

## 67: neu:neu_base_02344_sad

Cohort neu_emotional; family family_food_23; intended sad

USER: I’ll pass on the table for three and go with a couple of seats instead.


A:

I hear you're ready to move forward with a different seating arrangement, and I'm glad you're comfortable making that choice.

B:

I hear that you're feeling a bit down about the table, and I'm here to support you as you make that choice.

## 68: neu:neu_base_02347_angry

Cohort neu_emotional; family family_food_23; intended angry

USER: I need a table for three and have been waiting for twenty minutes without a response.


A:

I understand you've been waiting for a response for 20 minutes, and I want to acknowledge that patience can feel frustrating. Please let me know if you need anything specific or if you'd like to discuss what's been going on.

B:

I understand how frustrating it is to wait so long for a response, and I'm sorry you're dealing with this delay.

## 69: neu:neu_base_02347_sad

Cohort neu_emotional; family family_food_23; intended sad

USER: I need a table for three and have been waiting for twenty minutes without a response.


A:

I understand how frustrating it must be to wait that long, and I'm sorry you're dealing with this delay.

B:

I understand you've been waiting for a response for 20 minutes, and I want to acknowledge that patience can feel frustrating. Please let me know if you need anything specific or if you'd like to discuss what's been going on.

## 70: neu:neu_base_02349_fearful

Cohort neu_emotional; family family_food_23; intended fearful

USER: We need a table for three and would like to sit close to the entrance.


A:

I understand you are feeling a bit worried about the table, and I want to reassure you that you can sit comfortably there.

B:

I'm glad you're thinking about making the space feel welcoming for three people. Let's work together to create a table that fits your needs and ensures everyone can sit close to the entrance comfortably.

## 71: neu:neu_base_02349_happy

Cohort neu_emotional; family family_food_23; intended happy

USER: We need a table for three and would like to sit close to the entrance.


A:

I'd be happy to help you find a table for three near the entrance.

B:

I'm glad you're excited about the new table, and I'm happy to help you secure a spot right near the entrance.

## 72: neu:neu_base_02967_angry

Cohort neu_emotional; family family_study_29; intended angry

USER: I’m waiting for my partner to finish the final example in the worksheet before we move on.


A:

I can see you're focused on finishing that worksheet example, and I'm here to help you get through it when you're ready.

B:

I understand you're waiting for your partner to complete the final example, and I'm here to support you while you wait.

## 73: neu:neu_base_02967_sad

Cohort neu_emotional; family family_study_29; intended sad

USER: I’m waiting for my partner to finish the final example in the worksheet before we move on.


A:

I understand you're waiting for your partner to complete the final example, and I'm here to support you while you wait.

B:

I can see you're focused on finishing that worksheet example, and I'm here to help you get through it when you're ready.

## 74: neu:neu_base_03150_angry

Cohort neu_emotional; family family_friends_31; intended angry

USER: I need to borrow the book you had earlier this week.


A:

I'd be happy to lend you the book, though I'd love to know if you've already read it or if you'd like to see it again.

B:

I understand you need the book sooner than usual, and I will be happy to lend it to you this week.

## 75: neu:neu_base_03150_happy

Cohort neu_emotional; family family_friends_31; intended happy

USER: I need to borrow the book you had earlier this week.


A:

I'd be happy to lend you the book, though I'd love to know if you've already read it or if you'd like to see it again.

B:

I understand you need to borrow the book, and I'm happy to lend it to you this week.

## 76: neu:neu_base_03952_happy

Cohort neu_emotional; family family_friends_39; intended happy

USER: I think I left my phone at home, so I’ll bring it to the birthday gathering.


A:

I hear your concern about the phone, and I'm glad you have a plan to handle it. Please take your time to find it, and I'm here if you need any help locating it.

B:

I'm glad to hear you're thinking of bringing a gift to the gathering, even if you're leaving your phone behind.

## 77: neu:neu_base_03952_sad

Cohort neu_emotional; family family_friends_39; intended sad

USER: I think I left my phone at home, so I’ll bring it to the birthday gathering.


A:

I understand you're feeling a bit worried about losing your phone, and I'm here to help you plan a way to get it back to your home.

B:

I hear your concern about the phone, and I'm glad you have a plan to handle it. Please take your time to find it, and I'm here if you need any help locating it.

## 78: neu:neu_base_04107_angry

Cohort neu_emotional; family family_home_41; intended angry

USER: I'll wait until Monday to see if the plumber returns to fix the leaking kitchen tap properly.


A:

I understand your frustration with the leak, and I'm here to help you get that kitchen tap fixed by Monday.

B:

I understand you're taking a moment to wait for the plumber to return, and I'm here if you need to talk through the situation or just need a quiet space to think about it.

## 79: neu:neu_base_04107_sad

Cohort neu_emotional; family family_home_41; intended sad

USER: I'll wait until Monday to see if the plumber returns to fix the leaking kitchen tap properly.


A:

I understand your concern about the leaky tap, and I'm here to help you wait until Monday to get it fixed.

B:

I understand you're taking a moment to wait for the plumber to return, and I'm here if you need to talk through the situation or just need a quiet space to think about it.

## 80: neu:neu_base_04109_fearful

Cohort neu_emotional; family family_home_41; intended fearful

USER: I expect the plumber to fix the leaking kitchen tap by the end of this week, no later.


A:

I understand your concern about the leaky tap, and I will make sure to get it fixed by the end of this week.

B:

I understand you're eager for the leak to be fixed by the end of this week, and I'm here to help you track down the plumber who can get it done.

## 81: neu:neu_base_04109_happy

Cohort neu_emotional; family family_home_41; intended happy

USER: I expect the plumber to fix the leaking kitchen tap by the end of this week, no later.


A:

I hear your frustration about the delay, and I want to acknowledge how much you need this fixed by the end of the week.

B:

I'm glad to hear you're looking forward to having a clean kitchen this week!

## 82: neu:neu_base_04310_angry

Cohort neu_emotional; family family_work_43; intended angry

USER: I'll send the updated version of the shared spreadsheet by 5 PM today.


A:

I understand you're eager to get the updated spreadsheet to me by 3 PM, and I'll do my best to ensure it's ready for you then.

B:

I'm glad to hear you have the updated spreadsheet ready to share with me today.

## 83: neu:neu_base_04310_happy

Cohort neu_emotional; family family_work_43; intended happy

USER: I'll send the updated version of the shared spreadsheet by 5 PM today.


A:

I'm glad to hear you'll be sending the updated spreadsheet today; I'll be here whenever you're ready to share it.

B:

I'm glad to hear you're ready to share the updated spreadsheet by 5 PM today.

## 84: neu:neu_base_04475_angry

Cohort neu_emotional; family family_technology_44; intended angry

USER: Thanks for helping me find the login after I misplaced it.


A:

I'm glad I could help you locate your login, and I'm here if you need anything else.

B:

I'm glad I could help you locate your login, and I hope you're able to get back into your account soon.

## 85: neu:neu_base_04475_sad

Cohort neu_emotional; family family_technology_44; intended sad

USER: Thanks for helping me find the login after I misplaced it.


A:

I'm glad I could help you locate your login, and I'm here if you need anything else.

B:

I'm glad I could help you get back into your account, and I hope you're able to use it again without any more issues.

## 86: neu:neu_base_04476_happy

Cohort neu_emotional; family family_technology_44; intended happy

USER: The connection is now active, though I still can’t see the file list.


A:

I'm glad to hear the connection is active, and I'm here to help you find the file on your list.

B:

I hear your frustration with the missing file list, and I want to reassure you that the connection is indeed active, even if the view isn't showing yet. Please take a moment to wait, as this is a common occurrence that will resolve itself soon.

## 87: neu:neu_base_04476_sad

Cohort neu_emotional; family family_technology_44; intended sad

USER: The connection is now active, though I still can’t see the file list.


A:

I hear your frustration with the missing file list, and I want to reassure you that the connection is indeed active, even if the view isn't showing yet. Please take a moment to wait, as this is a common occurrence that will resolve itself soon.

B:

I understand that the connection is active, but I'm still unable to see the file list, so I'm here to help you find it.
