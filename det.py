def determine_note(note,standard_notes):

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    max_match_cnt=0
    note_kp, note_des = sift.detectAndCompute(note,None)
    for st_note in standard_notes:
        # find the keypoints and descriptors with SIFT
        st_note_kp, st_note_des = sift.detectAndCompute(st_note,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(note_des,st_note_des,k=2)

        #print("no of matches: ",len(matches))

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        match_cnt=0
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.5*n.distance:
                matchesMask[i]=[1,0]
                match_cnt+=1


        if match_cnt>max_match_cnt:
            max_match_cnt=match_cnt
            final_note=st_note
            final_matches_mask=matchesMask
            final_note_kp=st_note_kp
            final_note_des=st_note_des



    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = final_matches_mask,
                       flags = 0)

    matching_image = cv2.drawMatchesKnn(note,note_kp,final_note,final_note_kp,matches,None,**draw_params)
    show_images([matching_image])

    show_images([note,final_note],["unknown","found"])
    return final_note