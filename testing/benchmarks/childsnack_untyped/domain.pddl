;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; The child-snack domain 2013
;;
;; This domain is for planning how to make and serve sandwiches for a group of
;; children in which some are allergic to gluten. There are two actions for
;; making sandwiches from their ingredients. The first one makes a sandwich and
;; the second one makes a sandwich taking into account that all ingredients are
;; gluten-free. There are also actions to put a sandwich on a tray, to move a tray
;; from one place to another and to serve sandwiches.
;;
;; Problems in this domain define the ingredients to make sandwiches at the initial
;; state. Goals consist of having all kids served with a sandwich to which they
;; are not allergic.
;;
;; Author: Raquel Fuentetaja and Tomás de la Rosa
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(define (domain child-snack)
(:requirements :typing :equality)
;(:types child bread-portion content-portion sandwich tray place)
(:constants kitchen)

(:predicates (at_kitchen_bread ?b)
	     (at_kitchen_content ?c)
     	     (at_kitchen_sandwich ?s)
     	     (no_gluten_bread ?b)
       	     (no_gluten_content ?c)
      	     (ontray ?s ?t)
       	     (no_gluten_sandwich ?s)
	     (allergic_gluten ?c)
     	     (not_allergic_gluten ?c)
	     (served ?c)
	     (waiting ?c ?p)
             (at ?t ?p)
	     (notexist ?s)
             (child_t ?c)
             (bread_portion_t ?b)
             (content_portion_t ?c)
             (sandwich_t ?s)
             (tray_t ?t)
             (place_t ?p)
  )

(:action make_sandwich_no_gluten
	 :parameters (?s ?b ?c)
	 :precondition (and (sandwich_t ?s) (bread_portion_t ?b) (content_portion_t ?c)
                            (at_kitchen_bread ?b)
			    (at_kitchen_content ?c)
			    (no_gluten_bread ?b)
			    (no_gluten_content ?c)
			    (notexist ?s))
	 :effect (and
		   (not (at_kitchen_bread ?b))
		   (not (at_kitchen_content ?c))
		   (at_kitchen_sandwich ?s)
		   (no_gluten_sandwich ?s)
                   (not (notexist ?s))
		   ))


(:action make_sandwich
	 :parameters (?s ?b ?c)
	 :precondition (and (sandwich_t ?s) (bread_portion_t ?b) (content_portion_t ?c)
                            (at_kitchen_bread ?b)
			    (at_kitchen_content ?c)
                            (notexist ?s)
			    )
	 :effect (and
		   (not (at_kitchen_bread ?b))
		   (not (at_kitchen_content ?c))
		   (at_kitchen_sandwich ?s)
                   (not (notexist ?s))
		   ))


(:action put_on_tray
	 :parameters (?s ?t)
	 :precondition (and  (sandwich_t ?s) (tray_t ?t)
                             (at_kitchen_sandwich ?s)
			     (at ?t kitchen))
	 :effect (and
		   (not (at_kitchen_sandwich ?s))
		   (ontray ?s ?t)))


(:action serve_sandwich_no_gluten
 	:parameters (?s ?c ?t ?p)
	:precondition (and (sandwich_t ?s) (child_t ?c) (tray_t ?t) (place_t ?p)
		       (allergic_gluten ?c)
		       (ontray ?s ?t)
		       (waiting ?c ?p)
		       (no_gluten_sandwich ?s)
                       (at ?t ?p)
		       )
	:effect (and (not (ontray ?s ?t))
		     (served ?c)))

(:action serve_sandwich
	:parameters (?s ?c ?t ?p)
	:precondition (and (sandwich_t ?s) (child_t ?c) (tray_t ?t) (place_t ?p)
                           (not_allergic_gluten ?c)
	                   (waiting ?c ?p)
			   (ontray ?s ?t)
			   (at ?t ?p))
	:effect (and (not (ontray ?s ?t))
		     (served ?c)))

(:action move_tray
	 :parameters (?t ?p1 ?p2)
	 :precondition (and (tray_t ?t) (place_t ?p1) (place_t ?p2) (at ?t ?p1))
	 :effect (and (not (at ?t ?p1))
		      (at ?t ?p2)))


)
