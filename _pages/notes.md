---
layout: archive
permalink: /notes/
title: "Notes"
author_profile: true
header:
  image: "/images/notebook.jpg"
---

Collection of notes, mainly for my personal use. These posts are separated because they aren't as well-groomed as blog posts. Some might evolve into full blog posts over time, others will permanently remain in this less formal form.

{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% if post.is_note %}
      {% include archive-single.html %}
    {% endif %}
  {% endfor %}
{% endfor %}
