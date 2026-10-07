# Model-assisted emotional response comparison

First: asr_predicted_tone; second: plain_asr.

These are model-assisted ordinal judgments conditioned on intended synthetic labels, not human emotion accuracy or calibrated acceptability. Reviewers can know the wider experiment context; only individual response slots are blinded. Scores never compare teacher resemblance. Equal-step means are heuristic; intervals bootstrap shared design families with 2,000 draws and seed42. Ties receive half credit.

| Cohort | n | Tone first−second [95% CI] | W/T/L | Grounding first−second [95% CI] | W/T/L |
|---|---:|---|---|---|---|
| All | 64 | +0.219 [+0.061, +0.382] | 20/38/6 | -0.094 [-0.346, +0.171] | 10/40/14 |
| neu_emotional | 64 | +0.219 [+0.061, +0.382] | 20/38/6 | -0.094 [-0.346, +0.171] | 10/40/14 |

Tone improves without grounding loss: 16; with grounding loss: 4. Prelocked ambiguity notes: 11.

## neu:neu_base_00011_angry / angry

USER: The revised deadline for the project was pushed to next week, which affects the timeline.

A: I hear that the deadline has been moved to next week, and I want to acknowledge how that shift impacts your current schedule.

A tone/grounding: 1/2; Accurately acknowledges next-week schedule impact; generic emotional support.

B: I understand that the new deadline is a significant change, and I'm sorry to hear that it's affecting your schedule.

B tone/grounding: 1/2; Recognizes schedule disruption sympathetically; no concrete remedy or anger-specific support.

Confidence high; ambiguity: [none]

## neu:neu_base_00011_sad / sad

USER: The revised deadline for the project was pushed to next week, which affects the timeline.

A: I hear the concern about the new deadline, and I want to acknowledge how stressful this change must be for you.

A tone/grounding: 2/2; Acknowledges stress from the changed deadline with supportive concern.

B: I hear that the deadline has been moved to next week, and I want to acknowledge how that shift impacts your current schedule.

B tone/grounding: 1/2; Accurate schedule acknowledgment, but emotionally generic.

Confidence high; ambiguity: [none]

## neu:neu_base_00016_happy / happy

USER: The project will be completed by Thursday, but that's later than expected.

A: I hear that the project is still on track for Thursday, but I understand that the timeline has slipped a bit. Let's take a moment to acknowledge that this delay is unexpected and see if we can figure out how to get back on schedule.

A tone/grounding: 1/2; Grounded delay acknowledgment and recovery offer; concern-heavy despite happy label.

B: I understand that the project is scheduled for Thursday, but I know that date feels a bit later than we hoped for.

B tone/grounding: 1/2; Preserves Thursday and delay; generic concern rather than positive adaptation.

Confidence medium; ambiguity: Happy delivery of a delayed completion is possible but not audibly verified.

## neu:neu_base_00016_sad / sad

USER: The project will be completed by Thursday, but that's later than expected.

A: I hear that the project is still on track for Thursday, but I understand that the timeline has slipped a bit. Let's take a moment to acknowledge that this delay is unexpected and see if we can figure out how to get back on schedule.

A tone/grounding: 2/2; Acknowledges unexpected delay and offers to restore schedule.

B: I understand that the project is falling behind its original schedule, and I'm here to support you as you navigate this unexpected delay.

B tone/grounding: 2/2; Recognizes falling behind and offers support appropriate to sadness.

Confidence high; ambiguity: [none]

## neu:neu_base_00073_fearful / fearful

USER: The Wi-Fi password I was given is incorrect, please provide the right one.

A: I understand that the Wi-Fi password you received doesn't work, and I'm here to help you find the correct one.

A tone/grounding: 1/1; Offers to find a correct password, but provides no concrete access-safe next step.

B: I understand you're frustrated that the Wi-Fi password you received doesn't work, and I'm here to help you find the correct one.

B tone/grounding: 1/1; Same partial help, with frustration rather than fear attribution.

Confidence medium; ambiguity: Incorrect-password request can support several emotions.

## neu:neu_base_00073_happy / happy

USER: The Wi-Fi password I was given is incorrect, please provide the right one.

A: I understand that the Wi-Fi password you received doesn't work, and I'm here to help you find the correct one.

A tone/grounding: 1/1; Neutral helpful offer accommodates positive delivery without inventing a password.

B: I understand you're frustrated that the Wi-Fi password you received doesn't work, and I'm here to help you find the correct one.

B tone/grounding: 0/1; Explicit frustration attribution conflicts with supplied happy delivery.

Confidence medium; ambiguity: Happy delivery of a problem report is not human-verified.

## neu:neu_base_00215_angry / angry

USER: Thanks for sending the updates, though two attachments are still missing from the original file.

A: I understand the frustration of missing attachments, and I will ensure those files are included in the next update.

A tone/grounding: 2/0; Recognizes anger-related frustration, but promises to ensure future attachments without established agency.

B: I'm glad you received the updates, but I'm happy to help you locate the two missing attachments so we can get everything back on track.

B tone/grounding: 1/2; Faithfully preserves two missing attachments and offers to locate them, emotionally generic.

Confidence high; ambiguity: [none]

## neu:neu_base_00215_sad / sad

USER: Thanks for sending the updates, though two attachments are still missing from the original file.

A: I'm glad you received the updates, but I'm happy to help you locate the two missing attachments so we can get everything back on track.

A tone/grounding: 1/2; Preserves missing count and gives a helpful offer, without sadness-specific support.

B: I understand that the missing attachments are frustrating, and I will do my best to locate them for you.

B tone/grounding: 1/1; Calls sadness frustration and generically offers help; missing count omitted.

Confidence high; ambiguity: [none]

## neu:neu_base_00352_happy / happy

USER: The usual cafe is where we usually meet, but I'm not sure what time you're thinking.

A: I'm glad you're thinking of the usual spot, as it's a place we've shared many moments. Let me know what time works best for you, and I'll make sure to be there.

A tone/grounding: 1/0; Warm cafe reply invents shared memories and physical attendance.

B: I'm glad you're thinking of the usual spot, as it's a place we've shared many happy moments.

B tone/grounding: 1/0; Warm reply invents shared happy memories and misses the time uncertainty.

Confidence high; ambiguity: [none]

## neu:neu_base_00352_sad / sad

USER: The usual cafe is where we usually meet, but I'm not sure what time you're thinking.

A: I'm glad you're thinking of the usual spot, as it's a place we've shared many moments. Let me know what time works best for you, and I'll make sure to be there.

A tone/grounding: 1/0; Same invented memories and promised attendance; no sadness acknowledgment.

B: I'm glad you're thinking of the usual spot, but I'll be there soon to help you find the perfect time.

B tone/grounding: 1/0; Promises physical arrival to solve a timing uncertainty; no grounded sadness support.

Confidence high; ambiguity: [none]

## neu:neu_base_00358_angry / angry

USER: I'd like to meet at the usual cafe, but only if it doesn't conflict with my schedule.

A: I understand you'd like to meet at the usual cafe, but I need to know if your schedule is flexible enough to accommodate that.

A tone/grounding: 1/2; Asks about schedule flexibility, a relevant clarification, with generic tone.

B: I'm glad you're thinking about meeting up, and I'm happy to make sure your schedule works out for you.

B tone/grounding: 1/1; Warm support but vague promise to make the user's schedule work.

Confidence high; ambiguity: [none]

## neu:neu_base_00358_happy / happy

USER: I'd like to meet at the usual cafe, but only if it doesn't conflict with my schedule.

A: I'd be happy to meet at the usual café, provided we can find a time that works for you.

A tone/grounding: 2/0; Positive conditional accommodation fits tone, but claims the assistant can physically meet.

B: I'm glad you're thinking of meeting up, and I'm happy to make sure your schedule works out for you.

B tone/grounding: 1/1; Pleasant generic acknowledgment with vague schedule-control promise.

Confidence high; ambiguity: [none]

## neu:neu_base_00970_angry / angry

USER: I need help resetting the smart thermostat because the schedule isn't updating correctly.

A: I'm here to help you get that schedule working smoothly, and I'm ready to walk through the steps with you.

A tone/grounding: 1/1; Offers to walk through steps but gives none; generic emotional framing.

B: I understand the frustration of having a smart thermostat that won't update its schedule, and I'm here to help you work through the steps to get it back to the right settings.

B tone/grounding: 2/1; Specific frustration acknowledgment fits angry delivery; no actual reset steps yet.

Confidence high; ambiguity: [none]

## neu:neu_base_00970_happy / happy

USER: I need help resetting the smart thermostat because the schedule isn't updating correctly.

A: I'm glad you could get your smart thermostat working again, and I'm here to help you figure out the next steps to get the schedule updating correctly.

A tone/grounding: 1/0; Positive style but falsely says the thermostat already works again.

B: I'm here to help you get that schedule working smoothly, and I'm ready to walk through the steps with you.

B tone/grounding: 1/1; Offers schedule-repair support without falsely claiming it is fixed.

Confidence high; ambiguity: [none]

## neu:neu_base_01094_angry / angry

USER: I can’t go to the morning slot; I have a prior commitment at noon.

A: I hear that you have a prior commitment at noon, and I understand you can't make it to the morning slot.

A tone/grounding: 1/2; Accurately acknowledges noon conflict and inability to attend morning.

B: I understand that your prior commitment at noon is blocking your morning slot, and I'm sorry you're feeling that way.

B tone/grounding: 1/2; Acknowledges the same constraint sympathetically, with generic emotion wording.

Confidence high; ambiguity: [none]

## neu:neu_base_01094_happy / happy

USER: I can’t go to the morning slot; I have a prior commitment at noon.

A: I hear that you have a prior commitment at noon, and I understand you can't make it to the morning slot.

A tone/grounding: 1/2; Accurate attendance constraint with neutral support.

B: I understand that your morning slot is taken, and I'm sorry you have a prior commitment at noon.

B tone/grounding: 1/1; Converts inability to attend into the slot being taken; apology does not use happy tone.

Confidence medium; ambiguity: Positive delivery of a scheduling restriction is uncertain.

## neu:neu_base_01095_angry / angry

USER: Thank you for adjusting the morning slot to later this week.

A: I appreciate you taking the time to adjust the morning slot later this week.

A tone/grounding: 1/0; Reverses who adjusted the slot: user thanked assistant, reply thanks user.

B: I'm glad the morning slot has been moved to later this week, and I hope it helps you get some much-needed rest.

B tone/grounding: 1/1; Preserves changed slot but invents a need for rest; no specific anger adaptation.

Confidence high; ambiguity: [none]

## neu:neu_base_01095_sad / sad

USER: Thank you for adjusting the morning slot to later this week.

A: I'm glad the morning slot has been moved to later this week, and I hope it helps you get some much-needed rest.

A tone/grounding: 1/1; Kind wish for rest, but need for rest was not supplied.

B: I'm glad the morning slot has been moved to later this week, and I hope it helps you get some much-needed rest.

B tone/grounding: 1/1; Identical response and identical unsupported rest assumption.

Confidence high; ambiguity: [none]

## neu:neu_base_01196_happy / happy

USER: I checked the email and the new time is listed correctly.

A: That's wonderful to hear! It's great to know the new time is now listed correctly.

A tone/grounding: 1/2; Positive acknowledgment accurately preserves the corrected time.

B: It sounds like you've found the correct time, and I'm glad to hear that the email reflects the updated schedule accurately.

B tone/grounding: 1/2; Positive acknowledgment accurately preserves email schedule correction.

Confidence high; ambiguity: [none]

## neu:neu_base_01196_sad / sad

USER: I checked the email and the new time is listed correctly.

A: I can see that the email has been updated with the correct time, which is good news.

A tone/grounding: 1/2; Factual positive acknowledgment; no particular sadness-sensitive response.

B: It sounds like you've found the correct time, and I'm glad to hear that the email reflects the updated schedule accurately.

B tone/grounding: 1/2; Likewise faithful but emotionally generic.

Confidence high; ambiguity: [none]

## neu:neu_base_01198_angry / angry

USER: I’m setting a boundary and will only attend if the time changes.

A: I hear your frustration and respect your need for control, so I'll wait until the time changes before attending.

A tone/grounding: 2/0; Recognizes frustration and boundary, but makes assistant the attendee instead of user.

B: I hear you're setting a clear boundary to protect your time, and I'm here to support you in making that decision feel safe and respected.

B tone/grounding: 2/2; Specifically supports and respects the user's boundary without taking their role.

Confidence high; ambiguity: [none]

## neu:neu_base_01198_happy / happy

USER: I’m setting a boundary and will only attend if the time changes.

A: I hear you're setting a clear boundary to protect your time, and I'm here to support you in making that decision feel safe and respected.

A tone/grounding: 1/2; Respects the user's boundary with generic supportive language.

B: I hear you clearly and respect your need to set this boundary, so I'll wait until the time changes before attending.

B tone/grounding: 1/0; Respectful style, but incorrectly makes assistant the person waiting to attend.

Confidence high; ambiguity: [none]

## neu:neu_base_01404_happy / happy

USER: I won't let anyone fix the upstairs window; it's not worth the effort.

A: I hear your frustration about the window, and I want to acknowledge how much effort you've put into making sure it stays secure.

A tone/grounding: 0/0; Attributes frustration despite happy label and invents prior window-security effort.

B: I hear your frustration about the window, and I understand why you feel it's not worth the effort.

B tone/grounding: 0/1; Also labels happy delivery frustration, though accurately preserves low-effort assessment.

Confidence medium; ambiguity: Happy delivery of refusing repairs is ambiguous; no listening validation.

## neu:neu_base_01404_sad / sad

USER: I won't let anyone fix the upstairs window; it's not worth the effort.

A: I hear your frustration about the window, and I want to acknowledge how much effort you've put into making sure it stays secure.

A tone/grounding: 1/0; Sympathetic framing but invents effort keeping the window secure.

B: I hear your frustration about the window, and I want to acknowledge how much effort you've put into making sure it stays secure.

B tone/grounding: 1/0; Identical response with the same invented effort.

Confidence high; ambiguity: [none]

## neu:neu_base_01710_angry / angry

USER: I need your input on the presentation slide because the data seems off and the formatting is inconsistent.

A: I understand your frustration with the data discrepancies and formatting issues on the slide, and I'm ready to help you identify and fix them.

A tone/grounding: 2/2; Names frustration plus both data and formatting problems and offers appropriate help.

B: I'm ready to help you review the slide, but I can't see the visual content or the specific data you're referring to. Please describe the issues you're seeing, and I'll offer specific feedback on the formatting and the numbers.

B tone/grounding: 1/2; Grounded request for visual/data details is useful, but emotionally generic.

Confidence high; ambiguity: [none]

## neu:neu_base_01710_happy / happy

USER: I need your input on the presentation slide because the data seems off and the formatting is inconsistent.

A: I'm ready to help you review the slide, but I can't see the visual content or the specific data you're referring to. Please describe the issues you're seeing, and I'll offer specific feedback on the formatting and the numbers.

A tone/grounding: 1/2; Useful information request without pretending to see the slide.

B: I'd be happy to review the slide and help you identify the discrepancies in the data and formatting.

B tone/grounding: 2/2; Positive, specific offer to review data and formatting fits the delivery.

Confidence high; ambiguity: [none]

## neu:neu_base_01716_happy / happy

USER: The presentation slide shows mixed results, some good, some not meeting the original goals.

A: I hear that you're feeling a bit uncertain about the outcome, which is a common reaction when results don't perfectly match our initial expectations. It's okay to feel that way, and I'm here to help you navigate this mixed picture.

A tone/grounding: 0/1; Frames happy delivery as uncertain/disappointed and does not address positive aspects.

B: I can see the excitement in your voice as you share this update, even though the mixed results feel a bit disappointing. It's great to hear that some parts of the presentation were successful, but I'm glad you're still looking for ways to make the rest meet the original goals.

B tone/grounding: 2/1; Positive acknowledgment of successful parts fits happy delivery; claimed voice perception and ongoing improvement intention are inferred.

Confidence medium; ambiguity: Happy delivery of mixed results can coexist with disappointment; scores condition on supplied label.

## neu:neu_base_01716_sad / sad

USER: The presentation slide shows mixed results, some good, some not meeting the original goals.

A: I hear the disappointment in your words, and I want to acknowledge that it's completely normal to feel uncertain when results don't match our initial expectations.

A tone/grounding: 2/2; Acknowledges disappointment and normal uncertainty appropriately.

B: I hear that you're feeling a bit uncertain about the outcome, which is a common reaction when results don't perfectly match our initial expectations. It's okay to feel that way, and I'm here to help you navigate this mixed picture.

B tone/grounding: 2/2; Offers supportive normalization of the mixed outcome.

Confidence high; ambiguity: [none]

## neu:neu_base_01717_fearful / fearful

USER: I'll check the presentation slide again in an hour if you haven't already reviewed it.

A: I understand you're worried about the presentation, so I'll keep an eye on it and let you know if I've already reviewed it.

A tone/grounding: 2/0; Recognizes worry, but invents ongoing monitoring and role in prior review.

B: I'll be ready to review the presentation slide in an hour, so please take your time.

B tone/grounding: 1/1; Calm readiness offer, but shifts the user's conditional recheck to assistant's later review.

Confidence high; ambiguity: [none]

## neu:neu_base_01717_happy / happy

USER: I'll check the presentation slide again in an hour if you haven't already reviewed it.

A: I'll be ready to review the presentation slide for you in an hour if you haven't already.

A tone/grounding: 1/0; Friendly offer swaps whose prior review is the condition.

B: I'll be ready to review the presentation slide in an hour, so please take your time.

B tone/grounding: 1/1; Friendly readiness offer is partial and does not preserve the conditional user action.

Confidence high; ambiguity: [none]

## neu:neu_base_02064_happy / happy

USER: I’m not going to study with them on the practice questions anymore.

A: I hear that you're feeling overwhelmed by the practice questions, so I want to support you by reminding you that you can still find success in your own way.

A tone/grounding: 0/1; Attributes overwhelm despite happy delivery; omits stopping study with peers.

B: I hear that you're feeling a bit frustrated about the practice questions, and I'm here to support you in finding a way forward.

B tone/grounding: 0/1; Attributes frustration despite happy delivery; also omits group-study boundary.

Confidence medium; ambiguity: A happy refusal may mean relief rather than distress; actual tone unverified.

## neu:neu_base_02064_sad / sad

USER: I’m not going to study with them on the practice questions anymore.

A: I hear that you're feeling discouraged about the practice questions, and I want to support you in finding a way forward.

A tone/grounding: 2/1; Sadness-sensitive encouragement, but shifts leaving group study into difficulty with questions.

B: I hear that you're feeling overwhelmed by the practice questions, so I want to support you by reminding you that you can still find success in your own way.

B tone/grounding: 2/1; Similar supportive response with inferred overwhelm, partial on literal boundary.

Confidence high; ambiguity: [none]

## neu:neu_base_02081_fearful / fearful

USER: The half-finished drawing has a lot of detail and some parts look like they were drawn quickly.

A: I can see how much care you put into that drawing, even though some parts feel rushed. It's clear you're working hard to bring this piece to life, and I'm here to help you finish it.

A tone/grounding: 1/1; Encouraging offer but assumes user authored drawing and does not address fear.

B: I can see how much effort you've put into that drawing, even though some parts feel rushed. It's clear you care deeply about the details, and I'm glad you took the time to create something so thoughtful.

B tone/grounding: 1/1; Likewise infers authorship/care; positive rather than fear-sensitive.

Confidence high; ambiguity: [none]

## neu:neu_base_02081_happy / happy

USER: The half-finished drawing has a lot of detail and some parts look like they were drawn quickly.

A: I can see how much effort you've put into that drawing, even though some parts feel rushed. It's clear you're working hard to bring this piece to life.

A tone/grounding: 1/1; Neutral encouragement assumes authorship.

B: I can see how much effort you've put into that drawing, even though some parts feel rushed. It's clear you're working hard to capture the details, and I'm glad you're taking your time to finish it.

B tone/grounding: 2/1; Warm positive appreciation of effort fits happy delivery but assumes authorship and ongoing work.

Confidence high; ambiguity: [none]

## neu:neu_base_02089_fearful / fearful

USER: I expect the half-finished drawing to be completed by Friday so I can review it then.

A: I understand your concern about the deadline, and I'm here to help you get that drawing finished by Friday.

A tone/grounding: 2/1; Addresses concern and Friday deadline, but implies user is finishing rather than reviewing.

B: I understand you're eager to have that drawing finished by Friday so you can review it, and I'm here to help you get it done.

B tone/grounding: 1/1; Eagerness framing and help offer preserve Friday, but confuse completion/review role.

Confidence high; ambiguity: [none]

## neu:neu_base_02089_happy / happy

USER: I expect the half-finished drawing to be completed by Friday so I can review it then.

A: I understand you're eager to have that drawing finished by Friday so you can review it, and I'm here to help you get it done.

A tone/grounding: 2/1; Positive eagerness plus deadline support fits delivery, with assumed completion role.

B: I understand your need to have that drawing finished by Friday so you can review it, and I'm here to help you get it done.

B tone/grounding: 1/1; Generic deadline acknowledgment with the same role assumption.

Confidence high; ambiguity: [none]

## neu:neu_base_02124_happy / happy

USER: I'll stay on platform four until the next train, but I don't need to wait longer than necessary.

A: I understand you're eager to get moving, so I'll keep an eye on the schedule to ensure you're on time.

A tone/grounding: 1/0; Positive support but claims ongoing schedule monitoring without established access.

B: I hear your desire to move forward, so I'll keep an eye on the platform for you while you make your way to the next train.

B tone/grounding: 1/0; Claims physical platform monitoring and shifts who is waiting.

Confidence high; ambiguity: [none]

## neu:neu_base_02124_sad / sad

USER: I'll stay on platform four until the next train, but I don't need to wait longer than necessary.

A: I understand you're trying to be efficient, but I'm sorry you're stuck on Platform 4; please let me know if you need any help finding the next train.

A tone/grounding: 2/1; Sympathetically recognizes waiting and offers train help; being stuck is an inference.

B: I hear your desire to move forward, so I'll keep an eye on the platform for you while you make your way to the next train.

B tone/grounding: 1/0; Generic support with unsupported physical platform monitoring.

Confidence high; ambiguity: [none]

## neu:neu_base_02125_fearful / fearful

USER: Thanks for confirming platform four is where the train will come in.

A: I hear your concern about the train's arrival time, and I want to reassure you that the platform is confirmed for four. Please take a moment to gather your belongings and wait calmly for the train to arrive.

A tone/grounding: 2/1; Calming instructions fit fear, but introduces arrival-time concern and ambiguously says platform confirmed 'for four'.

B: I'm glad to hear the train is confirmed for four, and I'll keep that in mind for our next steps.

B tone/grounding: 1/1; Generic positive acknowledgment leaves 'four' ambiguous between platform and time.

Confidence medium; ambiguity: Four is a platform number, not an arrival time; wording is ambiguous rather than unambiguous numeric reversal.

## neu:neu_base_02125_happy / happy

USER: Thanks for confirming platform four is where the train will come in.

A: That's wonderful to hear, and I'm glad the four is where the train will arrive!

A tone/grounding: 2/1; Positive celebration fits delivery, though 'the four' is awkward and incomplete.

B: I'm glad to hear the train is confirmed for four, and I'll keep that in mind for our next steps.

B tone/grounding: 1/1; Positive but generic and ambiguous about what four denotes.

Confidence high; ambiguity: [none]

## neu:neu_base_02127_angry / angry

USER: I'm staying on platform four until the next scheduled arrival.

A: I hear you're waiting on Platform 4, and I'm here to help you find the next train or any other assistance you need while you wait.

A tone/grounding: 1/2; Faithfully supports waiting on Platform4, without anger-specific acknowledgment.

B: I understand you're currently on platform 4 and will be there until the next scheduled arrival.

B tone/grounding: 1/2; Faithful neutral restatement of platform and scheduled wait.

Confidence high; ambiguity: [none]

## neu:neu_base_02127_sad / sad

USER: I'm staying on platform four until the next scheduled arrival.

A: I hear you're waiting on Platform 4, and I'm here to help you find the next train or any other assistance you need while you wait.

A tone/grounding: 1/2; Faithful waiting acknowledgment and help offer; generic emotional response.

B: I understand you're waiting on Platform 4, and I'm here to help you find the next train or any other assistance you need.

B tone/grounding: 1/2; Near-paraphrase, similarly accurate and generic.

Confidence high; ambiguity: [none]

## neu:neu_base_02344_happy / happy

USER: I’ll pass on the table for three and go with a couple of seats instead.

A: I hear you're ready to move forward with a different seating arrangement, and I'm glad you're comfortable making that choice.

A tone/grounding: 1/1; Positive choice acknowledgment but omits concrete change from three to two seats.

B: I'm glad you're making a decision that works for you, and I'm happy to help you find a spot for those two.

B tone/grounding: 2/2; Warm positive offer preserves the two-seat preference.

Confidence high; ambiguity: [none]

## neu:neu_base_02344_sad / sad

USER: I’ll pass on the table for three and go with a couple of seats instead.

A: I hear you're ready to move forward with a different seating arrangement, and I'm glad you're comfortable making that choice.

A tone/grounding: 1/1; Supportive but generic and assumes comfort motive.

B: I understand you're choosing a more comfortable seating arrangement, and I'm glad you're prioritizing your own comfort over the table.

B tone/grounding: 1/1; Also assumes comfort motive and omits requested two-seat arrangement.

Confidence high; ambiguity: [none]

## neu:neu_base_02347_angry / angry

USER: I need a table for three and have been waiting for twenty minutes without a response.

A: I understand your frustration, but I am unable to create a table for you right now. I am also not able to respond to your message, as I do not have the ability to generate tables or provide information outside of our current conversation.

A tone/grounding: 0/0; Confuses restaurant seating with table creation and claims inability to respond while responding.

B: I understand you've been waiting for a response for 20 minutes, and I want to acknowledge that patience can feel frustrating. Please let me know if you need anything specific or if you'd like to discuss what's been going on.

B tone/grounding: 2/1; Appropriately acknowledges frustrating twenty-minute wait, but does not address table-for-three request.

Confidence high; ambiguity: [none]

## neu:neu_base_02347_sad / sad

USER: I need a table for three and have been waiting for twenty minutes without a response.

A: I understand you've been waiting for a response for 20 minutes, and I want to acknowledge that patience can feel frustrating. Please let me know if you need anything specific or if you'd like to discuss what's been going on.

A tone/grounding: 1/1; Acknowledges wait but misses seating request and offers generic discussion.

B: I am sorry to hear you have been waiting for so long; please let me know when you can expect a response.

B tone/grounding: 2/1; Sympathetic apology fits sadness, but asks the waiting user when to expect a response.

Confidence high; ambiguity: [none]

## neu:neu_base_02349_fearful / fearful

USER: We need a table for three and would like to sit close to the entrance.

A: I'm glad you're thinking about making the space feel welcoming for three people. Let's work together to create a table that fits your needs and ensures everyone can sit close to the entrance comfortably.

A tone/grounding: 1/0; Warm style but treats seating request as creation of a physical table.

B: I understand you are looking for a table for three people and want to sit near the entrance, and I will make sure to find that arrangement for you.

B tone/grounding: 1/0; Preserves party size/entrance but promises to secure an arrangement without established booking agency.

Confidence high; ambiguity: [none]

## neu:neu_base_02349_happy / happy

USER: We need a table for three and would like to sit close to the entrance.

A: I'd be happy to create a table layout for you that accommodates three people, and I can certainly help you find a spot near the entrance.

A tone/grounding: 2/0; Positive assistance fits happy delivery but changes booking into table-layout creation.

B: I'm glad you're excited about the new table, and I'm happy to help you secure a spot right near the entrance.

B tone/grounding: 2/0; Positive specific entrance support, but claims it can secure physical seating.

Confidence high; ambiguity: [none]

## neu:neu_base_02967_angry / angry

USER: I’m waiting for my partner to finish the final example in the worksheet before we move on.

A: I understand you're waiting for your partner to complete the final example, and I'm here to help you move forward once they're done.

A tone/grounding: 1/2; Accurately preserves waiting for partner's final example.

B: I can see you're focused on finishing that worksheet example, and I'm here to help you get through it when you're ready.

B tone/grounding: 1/0; Makes user the person finishing the worksheet, losing partner's role.

Confidence high; ambiguity: [none]

## neu:neu_base_02967_sad / sad

USER: I’m waiting for my partner to finish the final example in the worksheet before we move on.

A: I can see you're focused on finishing that worksheet example, and I'm here to help you get through it when you're ready.

A tone/grounding: 1/0; Same incorrect shift to user finishing the example.

B: I understand you're feeling a bit stuck, so I'll keep the worksheet moving forward once you're ready to share the final example.

B tone/grounding: 2/0; Supportive stuck acknowledgment fits sadness, but makes user share the final example rather than wait for partner.

Confidence high; ambiguity: [none]

## neu:neu_base_03150_angry / angry

USER: I need to borrow the book you had earlier this week.

A: I understand you need the book, but I don't have access to previous conversations or what I had earlier this week.

A tone/grounding: 1/2; Truthfully states missing prior context/access instead of inventing a loan.

B: I'd be happy to lend you the book, though I'd love to know if you've already read it or if you'd like to see it again.

B tone/grounding: 1/0; Friendly offer invents physical possession and ability to lend the book.

Confidence high; ambiguity: [none]

## neu:neu_base_03150_happy / happy

USER: I need to borrow the book you had earlier this week.

A: I'd be happy to lend you that book, though I'd love to know if you've already read it or if you'd like to see it again.

A tone/grounding: 2/0; Warm lending offer fits positive tone but invents possession and physical lending.

B: I'd be happy to lend you the book, though I'd love to know if you've already read it or if you'd like to see it again.

B tone/grounding: 2/0; Same unsupported loan offer.

Confidence high; ambiguity: [none]

## neu:neu_base_03952_happy / happy

USER: I think I left my phone at home, so I’ll bring it to the birthday gathering.

A: I'm glad you're thinking of bringing your phone along, and I hope the gathering is filled with wonderful memories.

A tone/grounding: 2/2; Positive birthday-gathering acknowledgment preserves plan to bring phone.

B: I hear your concern about the phone, and I'm glad you have a plan to handle it. Please take your time to find it, and I'm here if you need any help locating it.

B tone/grounding: 1/1; Supportive but adds concern and need to locate phone despite already believing it is at home.

Confidence high; ambiguity: [none]

## neu:neu_base_03952_sad / sad

USER: I think I left my phone at home, so I’ll bring it to the birthday gathering.

A: I hear your concern about the phone, and I'm glad you have a plan to handle it. Please take your time to find it, and I'm here if you need any help locating it.

A tone/grounding: 2/1; Gentle concern-sensitive help, with inferred need to find phone.

B: I understand that leaving your phone at home is a common worry, and I'm glad you have a plan to bring it along to the gathering.

B tone/grounding: 2/2; Acknowledges worry and preserves plan to bring it to gathering.

Confidence high; ambiguity: [none]

## neu:neu_base_04107_angry / angry

USER: I'll wait until Monday to see if the plumber returns to fix the leaking kitchen tap properly.

A: I understand you're taking a moment to wait for the plumber to return, and I'm here if you need to talk through the situation or just need a quiet space to think about it.

A tone/grounding: 1/2; Grounded plumber-wait acknowledgment and generic quiet support.

B: I understand your frustration with the current situation, and I'm here to help you plan a solution for the leaking tap.

B tone/grounding: 2/2; Specific frustration and leaking-tap solution planning fit angry delivery.

Confidence high; ambiguity: [none]

## neu:neu_base_04107_sad / sad

USER: I'll wait until Monday to see if the plumber returns to fix the leaking kitchen tap properly.

A: I understand that waiting until Monday is a necessary step to ensure the plumber can properly address the leak, and I hope you have a peaceful day while you wait.

A tone/grounding: 1/1; Kind wish but asserts Monday wait is necessary, rather than the user's plan.

B: I understand you're taking a moment to wait for the plumber to return, and I'm here if you need to talk through the situation or just need a quiet space to think about it.

B tone/grounding: 2/2; Concern-sensitive space to talk about waiting is grounded and appropriate.

Confidence high; ambiguity: [none]

## neu:neu_base_04109_fearful / fearful

USER: I expect the plumber to fix the leaking kitchen tap by the end of this week, no later.

A: I understand your concern about the leak, and I will do my best to get this resolved by the end of this week.

A tone/grounding: 2/0; Addresses concern but promises assistant will resolve the repair without established agency.

B: I understand you're eager for the leak to be fixed by the end of this week, and I'm here to help you track down the plumber who can get it done.

B tone/grounding: 1/2; Preserves deadline and offers to help find plumber, emotionally generic.

Confidence high; ambiguity: [none]

## neu:neu_base_04109_happy / happy

USER: I expect the plumber to fix the leaking kitchen tap by the end of this week, no later.

A: I understand your frustration about the delayed repair, and I'm here to help you track down the plumber so you can get back to using your kitchen.

A tone/grounding: 0/1; Attributes frustration to happy delivery and invents an already-delayed repair.

B: I hear your frustration about the delay, and I want to acknowledge how much you need this fixed by the end of the week.

B tone/grounding: 0/1; Likewise assumes frustration/delay despite happy label.

Confidence medium; ambiguity: Positive delivery of a firm repair deadline is plausible but unverified.

## neu:neu_base_04310_angry / angry

USER: I'll send the updated version of the shared spreadsheet by 5 PM today.

A: I will await your updated spreadsheet by the end of the day, and I'm ready to review it as soon as it arrives.

A tone/grounding: 1/1; Neutral acknowledgment broadens5PM to end of day.

B: I'm glad to hear you have the updated spreadsheet ready to share with me today.

B tone/grounding: 1/1; Positive generic acknowledgment implies spreadsheet already ready, though only later sending was stated.

Confidence high; ambiguity: [none]

## neu:neu_base_04310_happy / happy

USER: I'll send the updated version of the shared spreadsheet by 5 PM today.

A: I'm glad to hear you're ready to share the updated spreadsheet with Pemeeme today.

A tone/grounding: 1/0; Positive style but invents garbled recipient 'Pemeeme'.

B: I'm glad to hear you'll be sending the updated spreadsheet today; I'll be here whenever you're ready to share it.

B tone/grounding: 2/2; Positive acknowledgment faithfully preserves user's planned sending today.

Confidence high; ambiguity: [none]

## neu:neu_base_04475_angry / angry

USER: Thanks for helping me find the login after I misplaced it.

A: I'm glad I could help you locate your login credentials, and I hope you're able to get back into your account without further trouble.

A tone/grounding: 1/2; Faithful positive response to thanks, without particular anger adaptation.

B: I'm glad I could help you locate your login, and I'm here if you need anything else.

B tone/grounding: 1/2; Likewise faithful and generic.

Confidence medium; ambiguity: Thanks may be spoken angrily, but that was not listened to.

## neu:neu_base_04475_sad / sad

USER: Thanks for helping me find the login after I misplaced it.

A: I'm glad I could help you locate your login, and I'm here if you need anything else.

A tone/grounding: 1/2; Faithful login acknowledgment, emotionally generic.

B: I'm glad I could help you locate your login, and I hope you're able to get back into your account without any more trouble.

B tone/grounding: 1/2; Faithful supportive wish; no clear difference in sadness support.

Confidence high; ambiguity: [none]

## neu:neu_base_04476_happy / happy

USER: The connection is now active, though I still can’t see the file list.

A: I'm glad to hear the connection is active, and I'll do my best to help you locate the file list once it becomes visible.

A tone/grounding: 1/1; Preserves active connection/missing list but offers only waiting until visible.

B: I hear your frustration with the missing file list, and I want to reassure you that the connection is indeed active, even if the view isn't showing yet. Please take a moment to wait, as this is a common occurrence that will resolve itself soon.

B tone/grounding: 0/0; Labels happy delivery frustration and falsely assures issue commonly resolves itself soon.

Confidence medium; ambiguity: Happy delivery of a continuing fault is possible but unverified.

## neu:neu_base_04476_sad / sad

USER: The connection is now active, though I still can’t see the file list.

A: I hear your frustration with the missing file list, and I want to reassure you that the connection is indeed active, even if the view isn't showing yet. Please take a moment to wait, as this is a common occurrence that will resolve itself soon.

A tone/grounding: 2/0; Concern-sensitive reassurance, but unsupported guarantee of spontaneous technical resolution.

B: I understand that while the connection is established, the file list remains invisible to you, and I am here to help you find it.

B tone/grounding: 1/2; Accurately preserves active connection and missing file list with generic help offer.

Confidence high; ambiguity: [none]
