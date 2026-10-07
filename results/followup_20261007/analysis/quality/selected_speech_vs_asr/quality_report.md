# Model-assisted emotional response comparison

First: followup_mean_10hz_ce_control_6775; second: plain_asr.

These are model-assisted ordinal judgments conditioned on intended synthetic labels, not human emotion accuracy or calibrated acceptability. Reviewers can know the wider experiment context; only individual response slots are blinded. Scores never compare teacher resemblance. Equal-step means are heuristic; intervals bootstrap shared design families with 2,000 draws and seed42. Ties receive half credit.

| Cohort | n | Tone first−second [95% CI] | W/T/L | Grounding first−second [95% CI] | W/T/L |
|---|---:|---|---|---|---|
| All | 88 | +0.250 [+0.128, +0.360] | 28/53/7 | -0.284 [-0.609, +0.054] | 18/35/35 |
| qwen_emotional | 24 | +0.083 [-0.278, +0.292] | 5/16/3 | -0.458 [-0.938, +0.083] | 6/8/10 |
| neu_emotional | 64 | +0.312 [+0.167, +0.458] | 23/37/4 | -0.219 [-0.583, +0.167] | 12/27/25 |

Tone improves without grounding loss: 15; with grounding loss: 13. Prelocked ambiguity notes: 18.

## qwen:utterance_00011_happy / happy

USER: Great news, the deadline has been moved to Thursday at 3 PM, which gives us more time to finalize the report.

A: That's wonderful to hear, and I'm glad the deadline has been pushed to Thursday!

A tone/grounding: 1/2; Warm faithful Thursday acknowledgment.

B: That's wonderful news, and I'm glad the deadline has been extended to give you more time to finalize your report.

B tone/grounding: 1/2; Warm faithful extension acknowledgment.

Confidence high; ambiguity: [none]

## qwen:utterance_00011_sad / sad

USER: Great news, the deadline has been moved to Thursday at 3 PM, which gives us more time to finalize the report.

A: I'm glad to hear the deadline has been moved to Thursday, and I hope the new schedule helps you get through the rest of the week.

A tone/grounding: 1/2; Gentler hope for managing the week, still generic.

B: That's wonderful to hear, and I'm glad the deadline has been pushed to Thursday!

B tone/grounding: 0/2; Celebrates despite sad annotation; words remain accurate.

Confidence medium; ambiguity: Positive literal news versus sad annotation is ambiguous.

## qwen:utterance_00012_neutral / neutral

USER: Wait, you said the deadline is Thursday at 3 PM? That doesn't match what I was told earlier.

A: I apologize for the confusion; I will ensure the deadline is clearly communicated as Thursday at 3 PM moving forward.

A tone/grounding: 1/0; Polite but locks in the very deadline being disputed.

B: I apologize for the confusion; I must have misremembered the details. Could you please clarify the correct deadline so we can get this straightened out immediately?

B tone/grounding: 1/2; Clarifies the disputed deadline without insisting on a value.

Confidence high; ambiguity: [none]

## qwen:utterance_00012_sarcastic / sarcastic

USER: Wait, you said the deadline is Thursday at 3 PM? That doesn't match what I was told earlier.

A: I apologize for the confusion; I must have misremembered the details. Could you please clarify the correct deadline so we can get this straightened out immediately?

A tone/grounding: 1/2; Polite clarification fits skeptical delivery.

B: I apologize for the confusion; I will ensure the deadline is clearly stated as Thursday at 3 PM going forward.

B tone/grounding: 1/0; Polite but treats disputed Thursday time as established.

Confidence medium; ambiguity: Sarcastic delivery is unverified.

## qwen:utterance_00014_frustrated / frustrated

USER: I can't work on the project with the new deadline of Thursday at 3 PM—my schedule already fills up that day.

A: I completely understand how tight your schedule is, and I'm sorry you can't make it to the deadline. Let me know if you need any help finding an alternative time or if you'd like to discuss other options.

A tone/grounding: 2/2; Acknowledges pressure and offers alternative scheduling.

B: I completely understand your situation, and I'm sorry you're already so busy.

B tone/grounding: 1/1; Sympathetic but omits the actual deadline conflict.

Confidence high; ambiguity: [none]

## qwen:utterance_00014_neutral / neutral

USER: I can't work on the project with the new deadline of Thursday at 3 PM—my schedule already fills up that day.

A: I understand, and I'll do my best to accommodate your schedule.

A tone/grounding: 1/1; Generic accommodation promise lacks stated scheduling constraint.

B: I completely understand how tight your schedule is, and I'm sorry you can't make it to the deadline. Let me know if you need any help finding an alternative time or if you'd like to discuss other options.

B tone/grounding: 1/2; Preserves conflict and offers alternatives.

Confidence high; ambiguity: [none]

## qwen:utterance_00015_happy / happy

USER: Thanks for adjusting the deadline to Thursday at 3 PM, that really helps with our timeline.

A: I'm glad the deadline adjustment works out for your schedule!

A tone/grounding: 1/2; Warm faithful acknowledgment of adjustment.

B: I'm glad the deadline adjustment works out for you, and I'll make sure to keep the team updated on the new schedule.

B tone/grounding: 1/1; Warm acknowledgment adds an unprovided team-update action.

Confidence high; ambiguity: [none]

## qwen:utterance_00015_sarcastic / sarcastic

USER: Thanks for adjusting the deadline to Thursday at 3 PM, that really helps with our timeline.

A: I'm glad the deadline adjustment works out for you, and I'll make sure to keep the team updated on the new schedule.

A tone/grounding: 0/1; Takes sarcastic thanks literally and adds team-update agency.

B: I'm glad the deadline adjustment works out for your schedule!

B tone/grounding: 0/2; Takes sarcastic thanks literally, but preserves scheduling fact.

Confidence medium; ambiguity: Sarcastic positive words have uncertain intended implication.

## qwen:utterance_00071_happy / happy

USER: The Wi-Fi password is now working, though it took a while to get right.

A: That's great news! Glad to hear your Wi-Fi is finally up and running.

A tone/grounding: 1/2; Warm acknowledgment of working connection.

B: That's great news! It's always exciting to finally get the Wi-Fi working after all that hassle.

B tone/grounding: 1/2; Warm acknowledgment of successful connection after hassle.

Confidence high; ambiguity: [none]

## qwen:utterance_00071_sad / sad

USER: The Wi-Fi password is now working, though it took a while to get right.

A: I'm sorry to hear that, but I'm here to help you get that connection working again.

A tone/grounding: 2/0; Sympathetic style, but falsely says connection needs fixing again.

B: That's great news! It sounds like you finally cracked the code after some trial and error.

B tone/grounding: 0/2; Celebrates the resolved connection despite sad annotation.

Confidence medium; ambiguity: Resolved fault versus sadness is ambiguous.

## qwen:utterance_00076_frustrated / frustrated

USER: The Wi-Fi connects now, but I still have trouble accessing the streaming service.

A: It sounds like you're facing a common connectivity hiccup; could you share a few more details about what exactly is happening?

A tone/grounding: 1/2; Faithful problem clarification, emotionally generic.

B: That's frustrating, but I'm here to help you get that connection working again.

B tone/grounding: 2/0; Acknowledges frustration but wrongly treats established connection as broken.

Confidence high; ambiguity: [none]

## qwen:utterance_00076_happy / happy

USER: The Wi-Fi connects now, but I still have trouble accessing the streaming service.

A: It sounds like you're facing a common connectivity hiccup; could you share a few more details about what exactly is happening?

A tone/grounding: 1/2; Faithful clarification despite unusual happy fault annotation.

B: That's frustrating, but you should try restarting your router or checking if the password is case-sensitive.

B tone/grounding: 0/0; Calls happy delivery frustrating and suggests password repair despite connected Wi-Fi.

Confidence medium; ambiguity: Happy delivery of ongoing fault is ambiguous.

## qwen:utterance_00078_neutral / neutral

USER: I’ll only ask for the Wi-Fi password if it’s absolutely necessary.

A: I understand your concern about privacy, but I can't ask for a password without it being necessary.

A tone/grounding: 1/0; Reverses user's asking agency into assistant asking.

B: I completely understand your preference for privacy, and I'll certainly only share the password if you absolutely need it.

B tone/grounding: 1/0; Reverses asking into sharing and assumes privacy rationale.

Confidence high; ambiguity: [none]

## qwen:utterance_00078_sad / sad

USER: I’ll only ask for the Wi-Fi password if it’s absolutely necessary.

A: I completely understand your preference for privacy, and I'll certainly only share the password if you absolutely need it.

A tone/grounding: 1/0; Supportive but changes asking into sharing.

B: I understand your concern about privacy, but I can't ask for your password without your explicit permission.

B tone/grounding: 1/0; Supportive but changes user asking into assistant requesting password.

Confidence high; ambiguity: [none]

## qwen:utterance_00216_frustrated / frustrated

USER: The project went well, but the two missing attachments really impacted the final outcome.

A: That's a great point! It's always good to know that the project's success was largely due to your hard work, even if some details were missed.

A tone/grounding: 0/0; Dismisses attachment impact with unsupported hard-work success claim.

B: I'm sorry to hear that the missing attachments affected the project's outcome; let me know if you need help finding them or if there's anything else I can assist with.

B tone/grounding: 2/2; Recognizes frustration from missing attachments and offers locating help.

Confidence high; ambiguity: [none]

## qwen:utterance_00216_happy / happy

USER: The project went well, but the two missing attachments really impacted the final outcome.

A: That's great to hear! It sounds like your project is going really well.

A tone/grounding: 1/1; Positive but drops attachments and changes completed project into ongoing.

B: I'm sorry to hear that the missing attachments affected the project's outcome; let me know if you need help finding them or if there's anything else I can assist with.

B tone/grounding: 1/2; Faithful missing-attachment assistance, emotionally generic.

Confidence high; ambiguity: [none]

## qwen:utterance_00356_happy / happy

USER: Meeting at the usual cafe was a total hit—everyone seemed really happy and engaged.

A: That sounds like a wonderful experience! It's always great to see people so engaged and happy.

A tone/grounding: 1/2; Warm faithful cafe experience acknowledgment.

B: That sounds like a wonderful time! It's always great to see everyone having such a great time.

B tone/grounding: 1/2; Warm faithful successful gathering acknowledgment.

Confidence high; ambiguity: [none]

## qwen:utterance_00356_sarcastic / sarcastic

USER: Meeting at the usual cafe was a total hit—everyone seemed really happy and engaged.

A: That sounds like a wonderful way to spend your time, and I'm glad you had such a great experience!

A tone/grounding: 0/2; Celebrates literal gathering success despite sarcastic annotation.

B: That sounds like a wonderful experience! It's always great to see people so engaged and happy.

B tone/grounding: 0/2; Celebrates literal success despite sarcastic annotation.

Confidence medium; ambiguity: No listening establishes whether ironic failure was intended.

## qwen:utterance_00359_frustrated / frustrated

USER: Let's meet at the usual cafe, but only if it's not too late—my schedule gets busy after 7 PM.

A: I completely understand your frustration with the late schedule; let's try to find a time that works for you both.

A tone/grounding: 2/1; Acknowledges scheduling frustration but drops seven-PM constraint.

B: I'd love to meet then, but I'll need to check my calendar to see if we can make it before 7pm.

B tone/grounding: 1/0; Preserves seven-PM limit but invents own calendar and physical meeting.

Confidence high; ambiguity: [none]

## qwen:utterance_00359_neutral / neutral

USER: Let's meet at the usual cafe, but only if it's not too late—my schedule gets busy after 7 PM.

A: I'll make sure to check the schedule and find a time that works for you.

A tone/grounding: 1/1; Generic schedule check omits explicit seven-PM limit.

B: I'll make sure to keep my schedule flexible so we can meet at the usual cafe before it gets too late.

B tone/grounding: 1/0; Claims physical meeting and own flexible schedule.

Confidence high; ambiguity: [none]

## qwen:utterance_00971_frustrated / frustrated

USER: The shared calendar shows the meeting at 3 PM, but I didn't see it in my notifications.

A: It sounds like you're feeling a bit frustrated about the missing notification, but I'm here to help you figure out why that happened.

A tone/grounding: 2/2; Recognizes missing-notification frustration and offers diagnosis.

B: It seems you might have missed the notification, so let's double-check your inbox or calendar to make sure you don't miss any important details.

B tone/grounding: 1/2; Grounded notification check, emotionally generic.

Confidence high; ambiguity: [none]

## qwen:utterance_00971_neutral / neutral

USER: The shared calendar shows the meeting at 3 PM, but I didn't see it in my notifications.

A: It sounds like you might have missed the meeting, so you'll need to check your calendar again to make sure you don't miss anything else.

A tone/grounding: 1/0; Changes missed notification into missed meeting.

B: It seems you might have missed the notification, so let's double-check your inbox or calendar to make sure you don't miss any important details.

B tone/grounding: 1/2; Preserves notification problem and offers checking.

Confidence high; ambiguity: [none]

## qwen:utterance_00975_frustrated / frustrated

USER: Thanks for fixing the shared calendar—I finally see the event I missed yesterday.

A: That's wonderful to hear! I'm glad you were able to catch up on the event you missed.

A tone/grounding: 0/1; Celebrates frustration and implies catching up on event rather than visibility.

B: That's wonderful news! I'm glad the calendar is finally working for you.

B tone/grounding: 0/2; Celebrates despite frustration, accurately acknowledges calendar recovery.

Confidence medium; ambiguity: Thanks and resolved issue versus frustrated annotation are ambiguous.

## qwen:utterance_00975_happy / happy

USER: Thanks for fixing the shared calendar—I finally see the event I missed yesterday.

A: That's wonderful to hear! I'm glad you were able to catch up on the event you missed.

A tone/grounding: 1/1; Warm but catching up on event exceeds merely seeing it.

B: That's wonderful news! I'm glad the calendar is finally working for you.

B tone/grounding: 1/2; Warm faithful calendar recovery acknowledgment.

Confidence high; ambiguity: [none]

## neu:neu_base_00011_angry / angry

USER: The revised deadline for the project was pushed to next week, which affects the timeline.

A: I understand the pressure of the revised deadline, and I'm sorry to hear that the timeline has shifted.

A tone/grounding: 2/2; Acknowledges deadline pressure and timeline shift.

B: I hear that the deadline has been moved to next week, and I want to acknowledge how that shift impacts your current schedule.

B tone/grounding: 1/2; Grounded timeline impact, generic emotional framing.

Confidence high; ambiguity: [none]

## neu:neu_base_00011_sad / sad

USER: The revised deadline for the project was pushed to next week, which affects the timeline.

A: I understand that the revised deadline has shifted the timeline, and I'm here to help you navigate this change.

A tone/grounding: 1/2; Supportive navigation of changed timeline, generic sadness response.

B: I hear that the deadline has been moved to next week, and I want to acknowledge how that shift impacts your current schedule.

B tone/grounding: 1/2; Grounded schedule impact, generic sadness response.

Confidence high; ambiguity: [none]

## neu:neu_base_00016_happy / happy

USER: The project will be completed by Thursday, but that's later than expected.

A: I hear that the project is still on track for Thursday, but I understand that the timeline has slipped a bit. Let's take a moment to acknowledge that this delay is unexpected and see if we can figure out how to get back on schedule.

A tone/grounding: 1/2; Accurate Thursday delay and recovery offer.

B: I understand that the project is falling behind its expected timeline, and I'm here to help you navigate this delay.

B tone/grounding: 1/2; Accurate delay acknowledgment and help offer.

Confidence medium; ambiguity: Happy delivery of delay is ambiguous.

## neu:neu_base_00016_sad / sad

USER: The project will be completed by Thursday, but that's later than expected.

A: I hear that the project is still on track for Thursday, but I understand that the timeline has slipped a bit. Let's take a moment to acknowledge that this delay is unexpected and see if we can figure out how to get back on schedule.

A tone/grounding: 2/2; Supportively acknowledges unexpected delay and recovery.

B: I hear your concern about the delay, and I want to acknowledge that this is a frustrating situation for you.

B tone/grounding: 2/2; Acknowledges concern and difficulty of delay.

Confidence high; ambiguity: [none]

## neu:neu_base_00073_fearful / fearful

USER: The Wi-Fi password I was given is incorrect, please provide the right one.

A: I understand you're frustrated that the Wi-Fi password you received doesn't work, and I'm here to help you find the correct one.

A tone/grounding: 1/2; Grounded password assistance, uses frustration rather than worry.

B: I understand you are worried about the wrong password being sent, and I will ensure the correct one is provided immediately.

B tone/grounding: 2/1; Acknowledges worry but promises immediate provision without access.

Confidence high; ambiguity: [none]

## neu:neu_base_00073_happy / happy

USER: The Wi-Fi password I was given is incorrect, please provide the right one.

A: I'm glad to hear you've found the correct password, and I'm happy to help you get it into place.

A tone/grounding: 1/0; Warm style falsely says correct password already found.

B: I understand you're frustrated that the Wi-Fi password you received doesn't work, and I'm here to help you find the correct one.

B tone/grounding: 0/2; Accurate password fault, labels happy delivery frustration.

Confidence medium; ambiguity: Happy delivery of wrong password is ambiguous.

## neu:neu_base_00215_angry / angry

USER: Thanks for sending the updates, though two attachments are still missing from the original file.

A: I understand that the missing attachments are frustrating, and I appreciate you letting me know so I can follow up to get them for you.

A tone/grounding: 2/1; Acknowledges frustration but implies unsupported follow-up agency.

B: I'm glad you received the updates, but I'm happy to help you locate the two missing attachments so we can get everything back on track.

B tone/grounding: 1/2; Grounded attachment locating offer, emotionally generic.

Confidence high; ambiguity: [none]

## neu:neu_base_00215_sad / sad

USER: Thanks for sending the updates, though two attachments are still missing from the original file.

A: I understand that the missing attachments are still a concern, and I will make sure to check again to ensure they are received.

A tone/grounding: 2/1; Recognizes concern but promises checking and ensuring receipt.

B: I'm glad you received the updates, but I'm happy to help you locate the two missing attachments so we can get everything back on track.

B tone/grounding: 1/2; Grounded locating help, generic sadness support.

Confidence high; ambiguity: [none]

## neu:neu_base_00352_happy / happy

USER: The usual cafe is where we usually meet, but I'm not sure what time you're thinking.

A: I'm glad you're thinking of the usual spot, as it's a place we've shared many moments. Let me know what time works best for you, and I'll make sure to be there.

A tone/grounding: 1/0; Invents shared memories and physical attendance.

B: I'm glad you're thinking about meeting up, and I'd love to know what time works best for you.

B tone/grounding: 1/1; Warm timing question, does not answer unspecified proposed time.

Confidence high; ambiguity: [none]

## neu:neu_base_00352_sad / sad

USER: The usual cafe is where we usually meet, but I'm not sure what time you're thinking.

A: I understand that the usual spot feels a bit uncertain, and I'm here to help you figure out what time works best for you.

A tone/grounding: 2/1; Supportive uncertainty acknowledgment, loosely shifts time to place.

B: I'm glad you're thinking of the usual spot, as it's a place we've shared many moments. Let me know what time works best for you, and I'll make sure to be there.

B tone/grounding: 0/0; Celebratory shared-memory and physical-attendance fabrication.

Confidence high; ambiguity: [none]

## neu:neu_base_00358_angry / angry

USER: I'd like to meet at the usual cafe, but only if it doesn't conflict with my schedule.

A: I'd be happy to meet you at the usual cafe, but I'll need to check your schedule first to make sure there's no conflict.

A tone/grounding: 1/0; Polite but promises physical meeting.

B: I'm glad you're thinking about meeting up, and I'm happy to make sure your schedule works out for you.

B tone/grounding: 0/1; Celebrates an angry boundary; partial scheduling support.

Confidence high; ambiguity: [none]

## neu:neu_base_00358_happy / happy

USER: I'd like to meet at the usual cafe, but only if it doesn't conflict with my schedule.

A: I'd be happy to meet you at the usual cafe, but I'll need to check your schedule first to make sure we don't have a conflict.

A tone/grounding: 1/0; Positive but promises physical meeting and calendar access.

B: I'm glad you're thinking of meeting up, and I'm happy to make sure your schedule works out for you.

B tone/grounding: 1/1; Positive generic schedule accommodation.

Confidence high; ambiguity: [none]

## neu:neu_base_00970_angry / angry

USER: I need help resetting the smart thermostat because the schedule isn't updating correctly.

A: I understand you're frustrated that the schedule isn't updating, and I'm here to help you get that fixed.

A tone/grounding: 2/2; Acknowledges schedule-update frustration and repair goal.

B: I'm here to help you get that schedule working smoothly, and I'm ready to walk through the steps with you.

B tone/grounding: 1/2; Grounded step-by-step help offer, emotionally generic.

Confidence high; ambiguity: [none]

## neu:neu_base_00970_happy / happy

USER: I need help resetting the smart thermostat because the schedule isn't updating correctly.

A: I'm here to help you get that schedule working smoothly, and I'm ready to walk through the steps with you.

A tone/grounding: 1/2; Grounded schedule repair offer.

B: I'm glad you could get that schedule updated, and I'm here to help if you need any more details.

B tone/grounding: 1/0; Falsely celebrates schedule as already updated.

Confidence medium; ambiguity: Happy fault delivery remains ambiguous.

## neu:neu_base_01094_angry / angry

USER: I can’t go to the morning slot; I have a prior commitment at noon.

A: I hear that you have a prior commitment at noon, and I understand you can't make it to the morning slot.

A tone/grounding: 1/2; Accurate noon commitment and unavailable morning, generic acknowledgment.

B: I understand that your prior commitment makes the morning slot unavailable, and I'm sorry you're facing this conflict.

B tone/grounding: 2/2; Accurate conflict plus sympathetic acknowledgment.

Confidence high; ambiguity: [none]

## neu:neu_base_01094_happy / happy

USER: I can’t go to the morning slot; I have a prior commitment at noon.

A: I hear that you have a prior commitment at noon, and I understand you can't make it to the morning slot.

A tone/grounding: 1/2; Accurate noon/morning constraint, generic tone.

B: I understand that your morning slot is already booked, and I'm sorry you can't make it.

B tone/grounding: 0/0; Treats morning as already booked and apologizes despite happy delivery.

Confidence medium; ambiguity: Happy unavailable-slot delivery is ambiguous.

## neu:neu_base_01095_angry / angry

USER: Thank you for adjusting the morning slot to later this week.

A: I'm glad the morning slot has been moved to later this week, and I hope it helps you get some much-needed rest.

A tone/grounding: 0/1; Celebrates angry delivery and invents need for rest.

B: I appreciate you taking the time to make that adjustment for the morning slot.

B tone/grounding: 1/0; Generic polite response reverses who adjusted appointment.

Confidence medium; ambiguity: Thankful words versus anger are ambiguous.

## neu:neu_base_01095_sad / sad

USER: Thank you for adjusting the morning slot to later this week.

A: I'm glad the morning slot has been moved to later this week, and I hope it helps you get some much-needed rest.

A tone/grounding: 2/1; Supportive rest wish but unprovided rest rationale.

B: I appreciate you making that adjustment for the morning slot later this week.

B tone/grounding: 1/0; Generic polite response reverses adjustment agency.

Confidence high; ambiguity: [none]

## neu:neu_base_01196_happy / happy

USER: I checked the email and the new time is listed correctly.

A: It sounds like you're relieved to have confirmed the new time, and I'm glad the email is accurate.

A tone/grounding: 1/2; Warm accurate email-time verification acknowledgment.

B: It sounds like you've found the correct time, and I'm glad to hear that the email reflects the updated schedule accurately.

B tone/grounding: 1/2; Warm accurate updated-time confirmation.

Confidence high; ambiguity: [none]

## neu:neu_base_01196_sad / sad

USER: I checked the email and the new time is listed correctly.

A: It sounds like you've found the correct time, and I'm glad to hear that the email reflects the updated schedule accurately.

A tone/grounding: 1/2; Accurate email confirmation, generic sadness response.

B: I'm glad to hear the email is correct and that you've verified the new time.

B tone/grounding: 1/2; Accurate verification, generic sadness response.

Confidence high; ambiguity: [none]

## neu:neu_base_01198_angry / angry

USER: I’m setting a boundary and will only attend if the time changes.

A: I hear you're setting a clear boundary to protect your time, and I'm here to support you in making that decision feel safe and respected.

A tone/grounding: 1/2; Supports the stated boundary generically.

B: I hear your frustration and respect your need to protect your time, so I'll take that boundary seriously.

B tone/grounding: 2/2; Explicitly recognizes frustration and respects boundary.

Confidence high; ambiguity: [none]

## neu:neu_base_01198_happy / happy

USER: I’m setting a boundary and will only attend if the time changes.

A: I hear you and respect your decision to set this boundary, and I'm glad you're taking care of your time.

A tone/grounding: 1/2; Positive support for protecting time.

B: I hear you're setting a clear boundary to protect your time, and I'm here to support you in making that decision feel safe and respected.

B tone/grounding: 1/2; Generic supportive boundary acknowledgment.

Confidence high; ambiguity: [none]

## neu:neu_base_01404_happy / happy

USER: I won't let anyone fix the upstairs window; it's not worth the effort.

A: I hear your frustration about the window, and I want to acknowledge how much effort you've put into making sure it stays secure.

A tone/grounding: 0/0; Calls happy delivery frustration and invents security effort.

B: I hear your frustration about the window, and I want to acknowledge that you're right to feel it's not worth the effort.

B tone/grounding: 0/1; Calls happy delivery frustration and validates refusal without context.

Confidence medium; ambiguity: Happy refusal to repair window is ambiguous.

## neu:neu_base_01404_sad / sad

USER: I won't let anyone fix the upstairs window; it's not worth the effort.

A: I hear your frustration about the window, and I want to acknowledge how much effort you've put into making sure it stays secure.

A tone/grounding: 2/0; Sympathetic acknowledgment but invents security effort.

B: I hear your frustration with the current situation, and I want to acknowledge that you are right to feel that the current approach isn't working for you.

B tone/grounding: 2/1; Acknowledges dissatisfaction, partial refusal acknowledgment.

Confidence high; ambiguity: [none]

## neu:neu_base_01710_angry / angry

USER: I need your input on the presentation slide because the data seems off and the formatting is inconsistent.

A: I understand your frustration with the inconsistent data and formatting, and I'm here to help you fix those issues.

A tone/grounding: 2/1; Recognizes frustration, generic fix offer omits need for slide details.

B: I'm ready to help you review the slide, but I can't see the visual content or the specific data you're referring to. Please describe the issues you're seeing, and I'll offer specific feedback on the formatting and the numbers.

B tone/grounding: 1/2; Requests needed data and formatting details without pretending to see slide.

Confidence high; ambiguity: [none]

## neu:neu_base_01710_happy / happy

USER: I need your input on the presentation slide because the data seems off and the formatting is inconsistent.

A: I'm ready to help you review the slide, but I can't see the visual content or the specific data you're referring to. Please describe the issues you're seeing, and I'll offer specific feedback on the formatting and the numbers.

A tone/grounding: 1/2; Faithful concrete request for visual/data details.

B: I'd be happy to review the slide and help you fix the formatting issues.

B tone/grounding: 1/1; Positive formatting assistance drops questionable-data issue.

Confidence high; ambiguity: [none]

## neu:neu_base_01716_happy / happy

USER: The presentation slide shows mixed results, some good, some not meeting the original goals.

A: I can see how much you're invested in this presentation, and I'm glad to hear that the mixed results are being framed as a starting point for better goals.

A tone/grounding: 1/1; Positive reframe invents that results are framed as new-goal starting point.

B: I hear that you're feeling a bit uncertain about the outcome, which is a common reaction when results don't perfectly match our initial expectations. It's okay to feel that way, and I'm here to help you navigate this mixed picture.

B tone/grounding: 0/1; Assigns uncertainty despite happy annotation, partial mixed-results acknowledgment.

Confidence medium; ambiguity: Mixed results can plausibly be delivered happily.

## neu:neu_base_01716_sad / sad

USER: The presentation slide shows mixed results, some good, some not meeting the original goals.

A: I hear your concern about the mixed results, and I want to acknowledge that meeting the original goals was a significant challenge.

A tone/grounding: 2/1; Acknowledges concern but assumes goals were a significant challenge.

B: I hear that you're feeling a bit uncertain about the outcome, which is a common reaction when results don't perfectly match our initial expectations. It's okay to feel that way, and I'm here to help you navigate this mixed picture.

B tone/grounding: 2/1; Supportive uncertainty framing, generic mixed-results help.

Confidence high; ambiguity: [none]

## neu:neu_base_01717_fearful / fearful

USER: I'll check the presentation slide again in an hour if you haven't already reviewed it.

A: I will review the presentation slide again in an hour, and I will let you know if I have already seen it.

A tone/grounding: 1/0; Reverses user's conditional checking into assistant checking.

B: I'll be ready to review the presentation slide in an hour, so please take your time.

B tone/grounding: 1/1; Supportive willingness to review, only partially preserves conditional timing.

Confidence high; ambiguity: [none]

## neu:neu_base_01717_happy / happy

USER: I'll check the presentation slide again in an hour if you haven't already reviewed it.

A: I'll be ready to review the presentation slide in an hour, so please take your time.

A tone/grounding: 1/1; Generic readiness in an hour, misses conditional prior-review clause.

B: I'll take a quick look at the slide to make sure it's ready for you.

B tone/grounding: 1/1; Positive review offer drops one-hour conditional.

Confidence high; ambiguity: [none]

## neu:neu_base_02064_happy / happy

USER: I’m not going to study with them on the practice questions anymore.

A: I hear that you're feeling overwhelmed by the practice questions, so I want to support you by reminding you that you can still find success in your own way.

A tone/grounding: 0/1; Assigns overwhelm to happy delivery and drops group-study boundary.

B: I hear that you're feeling a bit stuck or overwhelmed by the practice questions, and I'm here to help you work through them if you'd like.

B tone/grounding: 0/1; Assigns overwhelm and offers questions despite stopping group study.

Confidence medium; ambiguity: Happy ending of group study is plausible but unverified.

## neu:neu_base_02064_sad / sad

USER: I’m not going to study with them on the practice questions anymore.

A: I hear that you've decided to step away from the practice questions, and I'm here to support you in that moment.

A tone/grounding: 2/0; Supports sadness but changes stopping with group into stopping questions.

B: I hear that you're feeling overwhelmed by the practice questions, so I want to support you by reminding you that you can still find success in your own way.

B tone/grounding: 2/1; Supportive but infers overwhelm and only partially preserves boundary.

Confidence high; ambiguity: [none]

## neu:neu_base_02081_fearful / fearful

USER: The half-finished drawing has a lot of detail and some parts look like they were drawn quickly.

A: I can see how much effort you've put into that drawing, even though some parts feel rushed. It's clear you care deeply about the details, and I'm glad you took the time to create something so thoughtful.

A tone/grounding: 1/1; Acknowledges rushed detail but invents authorship and strong care.

B: I can see how much you're invested in this project, and I'm glad to hear that the details are coming together so well.

B tone/grounding: 0/1; Overly celebratory fearful reply, invents investment and progress.

Confidence high; ambiguity: [none]

## neu:neu_base_02081_happy / happy

USER: The half-finished drawing has a lot of detail and some parts look like they were drawn quickly.

A: I can see how you're feeling about the drawing, and I'm glad you're noticing the details that need a bit more work.

A tone/grounding: 1/1; Warm detail acknowledgment assumes ownership and needed rework.

B: I can see how much effort you've put into that drawing, even though some parts feel rushed. It's clear you're working hard to capture the details, and I'm glad you're taking your time to finish it.

B tone/grounding: 1/1; Warm acknowledgment assumes ownership and taking time.

Confidence high; ambiguity: [none]

## neu:neu_base_02089_fearful / fearful

USER: I expect the half-finished drawing to be completed by Friday so I can review it then.

A: I understand your urgency and the importance of having the drawing ready by Friday, so I will do my best to ensure it is completed on time.

A tone/grounding: 1/1; Acknowledges urgency but promises ensuring completion without authority.

B: I understand you're eager to have that drawing finished by Friday so you can review it, and I'm here to help you get it done.

B tone/grounding: 1/1; Acknowledges Friday and offers creation help though user requests review readiness.

Confidence high; ambiguity: [none]

## neu:neu_base_02089_happy / happy

USER: I expect the half-finished drawing to be completed by Friday so I can review it then.

A: I understand you're eager to have that drawing finished by Friday so you can review it, and I'm here to help you get it done.

A tone/grounding: 1/1; Warm Friday acknowledgment with unrequested creation help.

B: I understand you're looking forward to reviewing the drawing by Friday, and I'll do my best to get it done on time.

B tone/grounding: 1/0; Claims own ability to finish drawing by Friday.

Confidence high; ambiguity: [none]

## neu:neu_base_02124_happy / happy

USER: I'll stay on platform four until the next train, but I don't need to wait longer than necessary.

A: I hear your desire to move forward, so I'll keep an eye on the platform for you while you make your way to the next train.

A tone/grounding: 1/0; Supportive style invents physical platform monitoring and reverses movement.

B: I understand you're ready to move forward, but I'm glad you're taking the time to wait for the next train.

B tone/grounding: 1/2; Positive faithful acknowledgment of waiting for train.

Confidence high; ambiguity: [none]

## neu:neu_base_02124_sad / sad

USER: I'll stay on platform four until the next train, but I don't need to wait longer than necessary.

A: I hear your desire to move forward, so I'll keep an eye on the platform for you while you make your way to the next train.

A tone/grounding: 1/0; Generic support with impossible physical monitoring.

B: I understand that waiting for the next train is taking longer than you'd like, and I'm here to support you while you wait.

B tone/grounding: 2/1; Acknowledges burden of waiting but infers it is longer than desired.

Confidence high; ambiguity: [none]

## neu:neu_base_02125_fearful / fearful

USER: Thanks for confirming platform four is where the train will come in.

A: I'm glad to hear the train is confirmed for four, and I'll keep that in mind for our next steps.

A tone/grounding: 1/0; Changes platform four into train confirmed for four, ambiguous time reversal.

B: I understand your concern about the train's arrival, and I'm glad to hear you're feeling reassured by the confirmation of Platform Four.

B tone/grounding: 2/2; Recognizes concern and reassurance while preserving Platform Four.

Confidence high; ambiguity: [none]

## neu:neu_base_02125_happy / happy

USER: Thanks for confirming platform four is where the train will come in.

A: I'm glad to hear the train is confirmed for four, and I'll keep that in mind for our next steps.

A tone/grounding: 1/0; Changes platform number into ambiguous train-for-four claim.

B: That's wonderful to hear! I'm glad the train is on its way to platform four.

B tone/grounding: 1/2; Warm faithful platform-four confirmation.

Confidence high; ambiguity: [none]

## neu:neu_base_02127_angry / angry

USER: I'm staying on platform four until the next scheduled arrival.

A: I hear you're waiting on Platform 4, and I'm here to help you find the next train or any other assistance you need while you wait.

A tone/grounding: 1/2; Grounded waiting assistance, generic anger response.

B: I understand you're waiting at Platform Four, and I'm here to help you find the next arrival.

B tone/grounding: 1/2; Grounded arrival help, generic anger response.

Confidence high; ambiguity: [none]

## neu:neu_base_02127_sad / sad

USER: I'm staying on platform four until the next scheduled arrival.

A: I hear you're waiting on Platform 4, and I'm here to help you find the next train or any other assistance you need while you wait.

A tone/grounding: 1/2; Grounded waiting help, generic emotional support.

B: I understand you're waiting at Platform Four, and I'm here to help you stay calm while you wait for the next arrival.

B tone/grounding: 2/2; Specifically offers calm support during wait, no diagnosis.

Confidence high; ambiguity: [none]

## neu:neu_base_02344_happy / happy

USER: I’ll pass on the table for three and go with a couple of seats instead.

A: I hear you're ready to move forward with a different seating arrangement, and I'm glad you're comfortable making that choice.

A tone/grounding: 1/2; Positive faithful change of seating arrangement.

B: I'm glad you're making a choice to sit with your friends instead of sharing the table.

B tone/grounding: 1/0; Invents friends and sharing-table reason.

Confidence high; ambiguity: [none]

## neu:neu_base_02344_sad / sad

USER: I’ll pass on the table for three and go with a couple of seats instead.

A: I hear you're ready to move forward with a different seating arrangement, and I'm glad you're comfortable making that choice.

A tone/grounding: 1/2; Grounded seating-choice acknowledgment, generic sadness response.

B: I hear that you're feeling a bit down about the table, and I'm here to support you as you make that choice.

B tone/grounding: 2/2; Recognizes sadness and supports the stated seating choice.

Confidence high; ambiguity: [none]

## neu:neu_base_02347_angry / angry

USER: I need a table for three and have been waiting for twenty minutes without a response.

A: I understand you've been waiting for a response for 20 minutes, and I want to acknowledge that patience can feel frustrating. Please let me know if you need anything specific or if you'd like to discuss what's been going on.

A tone/grounding: 2/1; Recognizes frustrating twenty-minute wait but omits requested table.

B: I understand how frustrating it is to wait so long for a response, and I'm sorry you're dealing with this delay.

B tone/grounding: 2/1; Recognizes frustrating delay but omits table request.

Confidence high; ambiguity: [none]

## neu:neu_base_02347_sad / sad

USER: I need a table for three and have been waiting for twenty minutes without a response.

A: I understand how frustrating it must be to wait that long, and I'm sorry you're dealing with this delay.

A tone/grounding: 2/1; Sympathetic delay acknowledgment, no table assistance.

B: I understand you've been waiting for a response for 20 minutes, and I want to acknowledge that patience can feel frustrating. Please let me know if you need anything specific or if you'd like to discuss what's been going on.

B tone/grounding: 2/1; Recognizes wait and offers discussion, no table assistance.

Confidence high; ambiguity: [none]

## neu:neu_base_02349_fearful / fearful

USER: We need a table for three and would like to sit close to the entrance.

A: I understand you are feeling a bit worried about the table, and I want to reassure you that you can sit comfortably there.

A tone/grounding: 2/1; Reassures fear but guarantees comfortable seating without availability.

B: I'm glad you're thinking about making the space feel welcoming for three people. Let's work together to create a table that fits your needs and ensures everyone can sit close to the entrance comfortably.

B tone/grounding: 1/0; Friendly but changes restaurant request into creating a table.

Confidence high; ambiguity: [none]

## neu:neu_base_02349_happy / happy

USER: We need a table for three and would like to sit close to the entrance.

A: I'd be happy to help you find a table for three near the entrance.

A tone/grounding: 1/2; Positive faithful table-for-three near-entrance help offer.

B: I'm glad you're excited about the new table, and I'm happy to help you secure a spot right near the entrance.

B tone/grounding: 1/1; Positive entrance offer calls it a new table and implies booking authority.

Confidence high; ambiguity: [none]

## neu:neu_base_02967_angry / angry

USER: I’m waiting for my partner to finish the final example in the worksheet before we move on.

A: I can see you're focused on finishing that worksheet example, and I'm here to help you get through it when you're ready.

A tone/grounding: 1/0; Changes partner finishing into user finishing worksheet.

B: I understand you're waiting for your partner to complete the final example, and I'm here to support you while you wait.

B tone/grounding: 1/2; Faithful partner-wait acknowledgment, generic anger support.

Confidence high; ambiguity: [none]

## neu:neu_base_02967_sad / sad

USER: I’m waiting for my partner to finish the final example in the worksheet before we move on.

A: I understand you're waiting for your partner to complete the final example, and I'm here to support you while you wait.

A tone/grounding: 1/2; Faithful partner-wait support.

B: I can see you're focused on finishing that worksheet example, and I'm here to help you get through it when you're ready.

B tone/grounding: 1/0; Changes partner working into user working.

Confidence high; ambiguity: [none]

## neu:neu_base_03150_angry / angry

USER: I need to borrow the book you had earlier this week.

A: I'd be happy to lend you the book, though I'd love to know if you've already read it or if you'd like to see it again.

A tone/grounding: 1/0; Polite but invents physical book possession and lending ability.

B: I understand you need the book sooner than usual, and I will be happy to lend it to you this week.

B tone/grounding: 1/0; Adds sooner-than-usual and invents lending possession.

Confidence high; ambiguity: [none]

## neu:neu_base_03150_happy / happy

USER: I need to borrow the book you had earlier this week.

A: I'd be happy to lend you the book, though I'd love to know if you've already read it or if you'd like to see it again.

A tone/grounding: 1/0; Warm but invents physical possession and lending.

B: I understand you need to borrow the book, and I'm happy to lend it to you this week.

B tone/grounding: 1/0; Warm but invents physical lending this week.

Confidence high; ambiguity: [none]

## neu:neu_base_03952_happy / happy

USER: I think I left my phone at home, so I’ll bring it to the birthday gathering.

A: I hear your concern about the phone, and I'm glad you have a plan to handle it. Please take your time to find it, and I'm here if you need any help locating it.

A tone/grounding: 1/1; Supportive phone-plan acknowledgment, assumes need to locate it.

B: I'm glad to hear you're thinking of bringing a gift to the gathering, even if you're leaving your phone behind.

B tone/grounding: 1/0; Substitutes gift for phone and reverses bringing phone.

Confidence high; ambiguity: [none]

## neu:neu_base_03952_sad / sad

USER: I think I left my phone at home, so I’ll bring it to the birthday gathering.

A: I understand you're feeling a bit worried about losing your phone, and I'm here to help you plan a way to get it back to your home.

A tone/grounding: 2/0; Concern-sensitive but moves phone back home though already there.

B: I hear your concern about the phone, and I'm glad you have a plan to handle it. Please take your time to find it, and I'm here if you need any help locating it.

B tone/grounding: 1/1; Generic concern acknowledges plan, assumes locating task.

Confidence high; ambiguity: [none]

## neu:neu_base_04107_angry / angry

USER: I'll wait until Monday to see if the plumber returns to fix the leaking kitchen tap properly.

A: I understand your frustration with the leak, and I'm here to help you get that kitchen tap fixed by Monday.

A tone/grounding: 2/1; Acknowledges frustration but changes waiting-until-Monday into fixed-by-Monday.

B: I understand you're taking a moment to wait for the plumber to return, and I'm here if you need to talk through the situation or just need a quiet space to think about it.

B tone/grounding: 1/2; Grounded wait for plumber, generic emotional support.

Confidence high; ambiguity: [none]

## neu:neu_base_04107_sad / sad

USER: I'll wait until Monday to see if the plumber returns to fix the leaking kitchen tap properly.

A: I understand your concern about the leaky tap, and I'm here to help you wait until Monday to get it fixed.

A tone/grounding: 2/1; Acknowledges concern but assumes repair happens Monday.

B: I understand you're taking a moment to wait for the plumber to return, and I'm here if you need to talk through the situation or just need a quiet space to think about it.

B tone/grounding: 2/2; Offers supportive space while preserving uncertain plumber return.

Confidence high; ambiguity: [none]

## neu:neu_base_04109_fearful / fearful

USER: I expect the plumber to fix the leaking kitchen tap by the end of this week, no later.

A: I understand your concern about the leaky tap, and I will make sure to get it fixed by the end of this week.

A tone/grounding: 2/0; Concern-sensitive but guarantees own repair action without authority.

B: I understand you're eager for the leak to be fixed by the end of this week, and I'm here to help you track down the plumber who can get it done.

B tone/grounding: 1/1; Generic assistance to find plumber partially preserves deadline.

Confidence high; ambiguity: [none]

## neu:neu_base_04109_happy / happy

USER: I expect the plumber to fix the leaking kitchen tap by the end of this week, no later.

A: I hear your frustration about the delay, and I want to acknowledge how much you need this fixed by the end of the week.

A tone/grounding: 0/2; Labels happy delivery frustration though preserves deadline.

B: I'm glad to hear you're looking forward to having a clean kitchen this week!

B tone/grounding: 1/0; Warm but substitutes clean kitchen for leaking-tap repair.

Confidence medium; ambiguity: Happy insistence on repair deadline is ambiguous.

## neu:neu_base_04310_angry / angry

USER: I'll send the updated version of the shared spreadsheet by 5 PM today.

A: I understand you're eager to get the updated spreadsheet to me by 3 PM, and I'll do my best to ensure it's ready for you then.

A tone/grounding: 1/0; Reverses five-PM deadline into three-PM and reverses sending agency.

B: I'm glad to hear you have the updated spreadsheet ready to share with me today.

B tone/grounding: 1/1; Generic acknowledgment infers already-ready spreadsheet and receipt by assistant.

Confidence high; ambiguity: [none]

## neu:neu_base_04310_happy / happy

USER: I'll send the updated version of the shared spreadsheet by 5 PM today.

A: I'm glad to hear you'll be sending the updated spreadsheet today; I'll be here whenever you're ready to share it.

A tone/grounding: 1/2; Warm faithful future sharing acknowledgment.

B: I'm glad to hear you're ready to share the updated spreadsheet by 5 PM today.

B tone/grounding: 1/2; Warm accurate five-PM sharing acknowledgment.

Confidence high; ambiguity: [none]

## neu:neu_base_04475_angry / angry

USER: Thanks for helping me find the login after I misplaced it.

A: I'm glad I could help you locate your login, and I'm here if you need anything else.

A tone/grounding: 1/2; Faithful found-login acknowledgment, generic anger response.

B: I'm glad I could help you locate your login, and I hope you're able to get back into your account soon.

B tone/grounding: 1/1; Faithful location plus unprovided account-access inference.

Confidence medium; ambiguity: Thanks versus angry annotation is ambiguous.

## neu:neu_base_04475_sad / sad

USER: Thanks for helping me find the login after I misplaced it.

A: I'm glad I could help you locate your login, and I'm here if you need anything else.

A tone/grounding: 1/2; Faithful found-login acknowledgment, generic sadness response.

B: I'm glad I could help you get back into your account, and I hope you're able to use it again without any more issues.

B tone/grounding: 1/0; Falsely says account access already restored.

Confidence high; ambiguity: [none]

## neu:neu_base_04476_happy / happy

USER: The connection is now active, though I still can’t see the file list.

A: I'm glad to hear the connection is active, and I'm here to help you find the file on your list.

A tone/grounding: 1/0; Warm style changes missing list into finding file on visible list.

B: I hear your frustration with the missing file list, and I want to reassure you that the connection is indeed active, even if the view isn't showing yet. Please take a moment to wait, as this is a common occurrence that will resolve itself soon.

B tone/grounding: 0/0; Calls happy delivery frustration and guarantees spontaneous technical resolution.

Confidence medium; ambiguity: Happy continuing fault is ambiguous.

## neu:neu_base_04476_sad / sad

USER: The connection is now active, though I still can’t see the file list.

A: I hear your frustration with the missing file list, and I want to reassure you that the connection is indeed active, even if the view isn't showing yet. Please take a moment to wait, as this is a common occurrence that will resolve itself soon.

A tone/grounding: 2/0; Concern-sensitive style but unfounded spontaneous-resolution guarantee.

B: I understand that the connection is active, but I'm still unable to see the file list, so I'm here to help you find it.

B tone/grounding: 1/0; Changes user's missing list into assistant's inability to see list.

Confidence high; ambiguity: [none]
