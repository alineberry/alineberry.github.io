import rss from '@astrojs/rss';
import { getEssays, postHref } from '../lib/posts';

export async function GET(context: { site?: URL }) {
  const essays = await getEssays();
  return rss({
    title: 'Adam Lineberry',
    description: 'Notes and writing on machine learning.',
    site: context.site!,
    items: essays.map((p) => ({
      title: p.data.title,
      pubDate: p.data.date,
      description: p.data.description,
      link: postHref(p),
    })),
  });
}
